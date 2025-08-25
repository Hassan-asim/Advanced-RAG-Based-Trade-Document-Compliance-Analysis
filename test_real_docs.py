#!/usr/bin/env python3
"""
Test script for improved document type detection using real documents
"""

import os
import sys
from rag_llm_pipeline import RAGLLMPipeline

def test_real_documents():
    """Test the improved document type detection with real document files"""
    
    # Initialize the pipeline
    pipeline = RAGLLMPipeline()
    
    # Test with actual document files
    doc_files = [
        ("DHL RECEIPT", "transcribe_docs/DHL RECEIPT.txt"),
        ("COMMERCIAL INVOICE", "transcribe_docs/COMMERCIAL INVOICE.txt"),
        ("BILL OF LADING", "transcribe_docs/BILL OF LADING.txt"),
        ("PACKING LIST", "transcribe_docs/PACKING LIST.txt"),
        ("SHIPMENT ADVICE", "transcribe_docs/SHIPMENT ADVICE.txt"),
        ("COVERING SCHEDULE", "transcribe_docs/COVERING SCHEDULE.txt")
    ]
    
    print("🧪 Testing Improved Document Type Detection with REAL Documents")
    print("=" * 70)
    
    for expected_type, file_path in doc_files:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        print(f"\n📄 Testing: {expected_type}")
        print(f"📁 File: {file_path}")
        print("-" * 50)
        
        try:
            # Read the actual file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📊 Content length: {len(content)} characters")
            
            # Test heuristic detection
            heuristic_result = pipeline._heuristic_detect_document_type(content)
            print(f"🔍 Heuristic Detection: {heuristic_result}")
            
            # Test full detection (with LLM fallback)
            full_result = pipeline.detect_document_type(content)
            print(f"🤖 Full Detection: {full_result}")
            
            # Check if detection was successful
            if heuristic_result == expected_type:
                print("✅ Heuristic detection: CORRECT")
            else:
                print(f"❌ Heuristic detection: INCORRECT (expected: {expected_type})")
                
            if full_result == expected_type:
                print("✅ Full detection: CORRECT")
            else:
                print(f"❌ Full detection: INCORRECT (expected: {expected_type})")
                
            # Show first 150 characters for context
            preview = content[:150].replace('\n', ' ').strip()
            print(f"📝 Preview: {preview}...")
            
            # Show key patterns found
            content_lower = content.lower()
            print("\n🔍 Key patterns found:")
            
            # Check for DHL patterns
            dhl_patterns = ["dhl", "waybill", "tracking", "airway", "express", "delivery", "parcel", "shipping", "consignment", "dispatch"]
            dhl_found = [p for p in dhl_patterns if p in content_lower]
            if dhl_found:
                print(f"  DHL indicators: {dhl_found}")
            
            # Check for invoice patterns
            invoice_patterns = ["commercial invoice", "invoice no", "invoice date", "total value", "cfr", "cif", "fob", "currency", "unit price", "quantity", "amount"]
            invoice_found = [p for p in invoice_patterns if p in content_lower]
            if invoice_found:
                print(f"  Invoice indicators: {invoice_found}")
            
            # Check for BOL patterns
            bol_patterns = ["bill of lading", "shipper", "exporter", "consignee", "notify party", "port of loading", "port of discharge", "vessel", "carrier", "container"]
            bol_found = [p for p in bol_patterns if p in content_lower]
            if bol_found:
                print(f"  BOL indicators: {bol_found}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    test_real_documents()
