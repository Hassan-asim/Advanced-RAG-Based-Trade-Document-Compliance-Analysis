import os
from dotenv import load_dotenv
from groq import Groq

# Import necessary libraries for PDF text extraction
import fitz
from PIL import Image
import pytesseract
import pdfplumber # New import

# Import for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the path to the Tesseract executable (removed hardcoded path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Removed

class PDFTranscriber:
    def __init__(self):
        load_dotenv()
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.groq_model = "llama3-70b-8192"
        self.input_pdf_dir = os.path.join("OCR", "share")
        self.output_transcribe_dir = "transcribe_docs"
        os.makedirs(self.output_transcribe_dir, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def _extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file, attempting pdfplumber first, then falling back to OCR.
        """
        text_content = ""
        try:
            # Attempt extraction with pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text with layout preservation
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text_content += page_text + "\n"
            if text_content.strip():
                print(f"  Extracted text from '{os.path.basename(pdf_path)}' using pdfplumber.")
                return text_content
            else:
                print(f"  pdfplumber extracted no text from '{os.path.basename(pdf_path)}'. Falling back to OCR.")

        except Exception as e:
            print(f"  Error with pdfplumber for '{os.path.basename(pdf_path)}': {e}. Falling back to OCR.")

        # Fallback to fitz + pytesseract OCR if pdfplumber fails or extracts no text
        try:
            document = fitz.open(pdf_path)
            text_content = ""
            # Changed PSM to 3 for general layout analysis
            tesseract_config = r'--psm 3'

            for page_num in range(document.page_count):
                page = document.load_page(page_num)
                text = page.get_text()

                if not text.strip():
                    print(f"  Page {page_num + 1} of '{os.path.basename(pdf_path)}' is image-based. Attempting OCR with preprocessing...")
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img = img.convert('L')
                    img = img.point(lambda x: 0 if x < 128 else 255, '1')
                    text = pytesseract.image_to_string(img, config=tesseract_config)

                text_content += text

            document.close()
            return text_content
        except pytesseract.TesseractNotFoundError:
            print(f"Error: Tesseract is not installed or not found in PATH. Please install it or ensure it's accessible.")
            return None
        except Exception as e:
            print(f"Error extracting text from '{pdf_path}': {e}")
            return None

    def transcribe_document(self, pdf_path):
        print(f"Transcribing {pdf_path}...")
        text_content = self._extract_text_from_pdf(pdf_path)
        if not text_content:
            print(f"Could not extract text from {pdf_path}")
            return

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text_content)
        transcribed_parts = []

        # Enhanced system prompt for better document formatting
        system_prompt = (
            "You are an expert document transcriber specializing in trade documents. Your task is to accurately transcribe "
            "document content while preserving the proper document structure and formatting.\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "1. Preserve title-value pairs where titles are above and values are below\n"
            "2. Maintain proper alignment and spacing between fields\n"
            "3. Keep the document structure with clear sections and headers\n"
            "4. Format addresses properly with line breaks and proper spacing\n"
            "5. Preserve numerical values, dates, reference codes, and amounts exactly\n"
            "6. Maintain the visual hierarchy and layout of the original document\n"
            "7. Preserve table structures and column alignments where possible\n"
            "8. Keep proper spacing between different document sections\n"
            "9. Do not add any summaries, interpretations, or conclusions\n"
            "10. Just provide the raw transcribed document text with proper formatting\n\n"
            "This document should maintain its original structure and formatting as much as possible."
        )

        for i, chunk in enumerate(text_chunks):
            print(f"  Processing chunk {i+1}/{len(text_chunks)} for {os.path.basename(pdf_path)}...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Transcribe the following document content with proper formatting:\n\n{chunk}"""}
            ]

            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.groq_model,
                    temperature=0.0,
                    max_tokens=4096,
                )
                transcribed_part = chat_completion.choices[0].message.content
                transcribed_parts.append(transcribed_part)
            except Exception as e:
                print(f"Error getting completion from Groq for chunk {i+1}: {e}")
                transcribed_parts.append(f"[TRANSCRIPTION FAILED FOR CHUNK {i+1}]")
                continue

        if transcribed_parts:
            final_transcribed_text = "".join(transcribed_parts)
            output_filename = os.path.basename(pdf_path).replace(".pdf", ".txt")
            output_path = os.path.join(self.output_transcribe_dir, output_filename)
            
            # Add a header based on document type
            doc_name = os.path.basename(pdf_path).replace(".pdf", "").upper()
            doc_header = f"=== {doc_name} TRANSCRIPTION ===\n\n"
            final_text_with_header = doc_header + final_transcribed_text
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text_with_header)
            print(f"Transcribed text saved to {output_path}")
        else:
            print(f"Failed to transcribe {pdf_path} (no parts transcribed).")

    def run(self):
        pdf_files = [f for f in os.listdir(self.input_pdf_dir) if f.endswith(".pdf")]
        for pdf_file in pdf_files:
            full_pdf_path = os.path.join(self.input_pdf_dir, pdf_file)
            self.transcribe_document(full_pdf_path)

if __name__ == "__main__":
    transcriber = PDFTranscriber()
    transcriber.run()
