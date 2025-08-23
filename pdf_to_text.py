import fitz  # PyMuPDF
import os
from PIL import Image
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convert_pdf_to_text(pdf_path, output_dir):
    """
    Converts a PDF file to a text file, using OCR if the PDF is image-based.
    Includes image preprocessing for more robust OCR.
    """
    try:
        document = fitz.open(pdf_path)
        text_content = ""
        
        # Tesseract configuration for better OCR
        # psm 6: Assume a single uniform block of text.
        # You can experiment with other PSMs (e.g., 3 for default, 1 for automatic page segmentation)
        tesseract_config = r'--psm 6'

        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            
            # Try to extract text directly
            text = page.get_text()
            
            if not text.strip(): # If direct text extraction is empty, try OCR
                print(f"  Page {page_num + 1} of '{os.path.basename(pdf_path)}' is image-based. Attempting OCR with preprocessing...")
                
                # Render page to a high-resolution image (pixmap)
                # Use a higher DPI (e.g., 300) for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72)) # Render at 300 DPI
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Image Preprocessing for OCR
                img = img.convert('L') # Convert to grayscale
                img = img.point(lambda x: 0 if x < 128 else 255, '1') # Binarization (black and white)
                
                text = pytesseract.image_to_string(img, config=tesseract_config)
            
            text_content += text

        document.close()

        # Create output file path
        pdf_filename = os.path.basename(pdf_path)
        text_filename = os.path.splitext(pdf_filename)[0] + ".txt"
        output_text_path = os.path.join(output_dir, text_filename)

        with open(output_text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_content)
        print(f"Successfully converted '{pdf_filename}' to '{text_filename}'")
    except pytesseract.TesseractNotFoundError:
        print(f"Error: Tesseract is not installed or not found in PATH. Please install it or set the correct path in the script.")
    except Exception as e:
        print(f"Error converting '{pdf_path}': {e}")

def process_pdfs_in_directory(input_dir):
    """
    Processes all PDF files in a given directory, converting them to text.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            convert_pdf_to_text(pdf_path, input_dir)

if __name__ == "__main__":
    # Define the share folder inside the OCR folder
    share_folder_path = r"C:\Users\walee\Desktop\1st task\OCR\share"
    process_pdfs_in_directory(share_folder_path)