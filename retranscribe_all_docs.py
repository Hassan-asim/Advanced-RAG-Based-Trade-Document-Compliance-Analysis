#!/usr/bin/env python3
"""
Script to re-transcribe all PDF documents from the share folder with improved formatting
"""

import os
import time
from PDF_transcriber import PDFTranscriber

def retranscribe_all_documents():
    """Re-transcribe all PDF documents from the share folder"""
    
    # Initialize the transcriber
    transcriber = PDFTranscriber()
    
    # Path to the share folder
    share_folder = os.path.join("OCR", "share")
    
    # Check if the share folder exists
    if not os.path.exists(share_folder):
        print(f"❌ Share folder not found at: {share_folder}")
        return
    
    # Get all PDF files in the share folder
    pdf_files = [f for f in os.listdir(share_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDF files found in the share folder")
        return
    
    print(f"🔍 Found {len(pdf_files)} PDF files to transcribe:")
    for pdf_file in pdf_files:
        print(f"  📄 {pdf_file}")
    
    print(f"\n🚀 Starting transcription of all documents...")
    print("=" * 60)
    
    # Track progress
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(share_folder, pdf_file)
        print(f"\n📋 Processing {i}/{len(pdf_files)}: {pdf_file}")
        print("-" * 50)
        
        try:
            # Transcribe the document
            transcriber.transcribe_document(pdf_path)
            
            # Check if transcription was successful
            output_filename = pdf_file.replace(".pdf", ".txt")
            output_path = os.path.join("transcribe_docs", output_filename)
            
            if os.path.exists(output_path):
                # Show file size
                file_size = os.path.getsize(output_path)
                print(f"✅ Successfully transcribed: {output_filename} ({file_size} bytes)")
                successful += 1
                
                # Show a small preview
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    preview = content[:200].replace('\n', ' ').strip()
                    print(f"📝 Preview: {preview}...")
            else:
                print(f"❌ Transcription failed for: {pdf_file}")
                failed += 1
                
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            failed += 1
        
        # Small delay between documents to avoid overwhelming the API
        if i < len(pdf_files):
            print("⏳ Waiting 2 seconds before next document...")
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRANSCRIPTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully transcribed: {successful} documents")
    print(f"❌ Failed: {failed} documents")
    print(f"📁 Total processed: {len(pdf_files)} documents")
    
    if successful > 0:
        print(f"\n📂 All transcribed documents saved to: transcribe_docs/")
        
        # List the output files
        print(f"\n📋 Transcribed files:")
        for pdf_file in pdf_files:
            txt_file = pdf_file.replace(".pdf", ".txt")
            txt_path = os.path.join("transcribe_docs", txt_file)
            if os.path.exists(txt_path):
                file_size = os.path.getsize(txt_path)
                print(f"  📄 {txt_file} ({file_size} bytes)")

if __name__ == "__main__":
    retranscribe_all_documents()
