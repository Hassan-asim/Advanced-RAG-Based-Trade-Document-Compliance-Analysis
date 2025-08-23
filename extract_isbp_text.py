import os
from PDF_transcriber import PDFTranscriber

pdf_path = 'C:\\Users\\walee\\Desktop\\1st task\\ISBP rules\\isbp-821-pr_2cafb582294ba9dfa9efc9a67b927758.pdf'
temp_output_file = 'C:\\Users\\walee\\Desktop\\1st task\\temp_isbp_821_text.txt'

transcriber = PDFTranscriber()
extracted_text = transcriber._extract_text_from_pdf(pdf_path)

if extracted_text:
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"Extracted text from {os.path.basename(pdf_path)} and saved to {temp_output_file}")
else:
    print(f"Failed to extract text from {os.path.basename(pdf_path)}")
