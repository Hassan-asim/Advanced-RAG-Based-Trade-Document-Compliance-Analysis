#!/usr/bin/env python3
"""
Script to transcribe the DHL RECEIPT PDF from the share folder with proper receipt formatting
"""

import os
from PDF_transcriber import PDFTranscriber

class ReceiptTranscriber(PDFTranscriber):
    def __init__(self):
        super().__init__()
    
    def transcribe_document(self, pdf_path):
        print(f"Transcribing {pdf_path} with receipt formatting...")
        text_content = self._extract_text_from_pdf(pdf_path)
        if not text_content:
            print(f"Could not extract text from {pdf_path}")
            return

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text_content)
        transcribed_parts = []

        # Special system prompt for receipt formatting
        system_prompt = (
            "You are an expert receipt transcriber. Your task is to accurately transcribe receipt documents "
            "while preserving the proper receipt format structure.\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "1. Preserve the title-value pairs where titles are above and values are below\n"
            "2. Maintain proper alignment and spacing between fields\n"
            "3. Keep the receipt structure with clear sections\n"
            "4. Format addresses properly with line breaks\n"
            "5. Preserve numerical values, dates, and reference codes exactly\n"
            "6. Maintain the visual hierarchy of the receipt\n"
            "7. Do not add any summaries or interpretations\n"
            "8. Just provide the raw transcribed receipt text with proper formatting\n\n"
            "This is a DHL receipt document that should maintain its receipt structure."
        )

        for i, chunk in enumerate(text_chunks):
            print(f"  Processing chunk {i+1}/{len(text_chunks)} for {os.path.basename(pdf_path)}...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Transcribe the following receipt content with proper formatting:\n\n{chunk}"""}
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
            
            # Add a header to indicate this is a receipt
            receipt_header = "=== DHL RECEIPT TRANSCRIPTION ===\n\n"
            final_text_with_header = receipt_header + final_transcribed_text
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text_with_header)
            print(f"Transcribed receipt text saved to {output_path}")
        else:
            print(f"Failed to transcribe {pdf_path} (no parts transcribed).")

def transcribe_dhl_receipt():
    """Transcribe only the DHL RECEIPT PDF with receipt formatting"""
    
    # Initialize the receipt transcriber
    transcriber = ReceiptTranscriber()
    
    # Path to the DHL RECEIPT PDF
    dhl_pdf_path = os.path.join("OCR", "share", "DHL RECEIPT.pdf")
    
    # Check if the PDF exists
    if not os.path.exists(dhl_pdf_path):
        print(f"❌ DHL RECEIPT PDF not found at: {dhl_pdf_path}")
        return
    
    print("🔍 Found DHL RECEIPT PDF, starting transcription with receipt formatting...")
    
    # Transcribe the document
    transcriber.transcribe_document(dhl_pdf_path)
    
    # Check if the transcription was successful
    output_path = os.path.join("transcribe_docs", "DHL RECEIPT.txt")
    if os.path.exists(output_path):
        print(f"✅ DHL RECEIPT successfully transcribed and saved to: {output_path}")
        
        # Show a preview of the transcribed content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"\n📄 Receipt transcription preview (first 500 characters):")
            print("=" * 60)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("=" * 60)
            
            # Show the file size
            file_size = os.path.getsize(output_path)
            print(f"📊 File size: {file_size} bytes")
    else:
        print("❌ Transcription failed - output file not created")

if __name__ == "__main__":
    transcribe_dhl_receipt()
