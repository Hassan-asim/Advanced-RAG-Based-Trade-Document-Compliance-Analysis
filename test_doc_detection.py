#!/usr/bin/env python3
"""
Test script for improved document type detection
"""

import os
import sys
from rag_llm_pipeline import RAGLLMPipeline

def test_document_type_detection():
    """Test the improved document type detection with sample documents"""
    
    # Initialize the pipeline
    pipeline = RAGLLMPipeline()
    
    # Test documents
    test_docs = {
        "DHL RECEIPT": """Here is the transcribed document content:

LE be 992@ 1000 C00! O60 arir)

EE

000000Zr+002r2Nd(12)

VOM) ARUN PUTT YOU MOAT

*¥ 910 & 9

ans 00'0 At
wna 00'O0 «snjeA swoysny juaueWwdg ecAl dxo/dwy
sjuawinoeg
3]U9U0D
bib by os'o |? os'0 SzO0z 20 82 OWAZOLOSZVELSB Spod jou
soaig «= |uBlapy juadiysebexoeg aep dnvoig ZS6LOOLZL ON Junoooy
!
aut Ff Aeg
SES)
lL
IH - Ad
IHOVUWy O0ZrL
WIN
GVO IHOV1OH IY
9018 AYYNOILVLS ONY ONILNIYd
V4 ON2 '(S0VeL) Nd
OSLINII YNVE GALINN +09
eu Ud
VISLSNy
Nala eraore
} ZEVIdCIIHOSHICH
YN
OV VINLSNY YNWE LIGZYOINN "HO
SIA °
NIDIYO
SIPHSAT
O «| JOIMGTYOM SSaydxX3a""",
        
        "COMMERCIAL INVOICE": """Starlinger
BULK FLEXIBLE PAKISTAN PVT LTD 1/2
501 5TH FLOOR BUSINESS AVENUE
SHAHRAH E FAISAL
KARACHI - PAKISTAN
Vienna, 2025 07 12
IBA
COMMERCIAL INVOICE No. MA4101597
Ref.: Irrevocable confirmed Documentary Credit (L/C No.) Number 0578ILC074195
Insurance covered by applicant.
Marine Cover Note No. 2025/04/CLFMIPDT00027 dated: 15.04.2025
H.S. Code No. 8446 2900 NTN No. 2679948-7
Job No. MA2053480
CIRCULAR LOOM RX 6.0 PRO (WITH STANDARD ACCESSORIES)
QTY: 4 UNITS AT RATE OF USD. 20,385 PER UNIT
PLUS FREIGHT CHARGES OF USD. 2,460/-
CFR KARACHI SEAPORT, PAKISTAN
ALL OTHER DETAILS AS PER PROFORMA INVOICE NO. PKGREIF-016.1
DATED: 08-APR-2025
TOTAL INVOICE VALUE CFR KARACHI SEAPORT, PAKISTAN USD 84.000,00
(acc. to ICC Incoterms 2020)
WE ARE HEREWITH CERTIFYING MERCHANDISE OF CHINA ORIGIN.
Port of Loading: Shanghai Seaport, China
Port of Discharge: Karachi Seaport, Pakistan""",
        
        "BILL OF LADING": """Here is the transcribed document content:

Ocean Track, Inc.

NVOCC License No. 15163N

SHIPPER/EXPORTER
STARLINGER Plastics Machinery (Taicang) Co. Ltd.
No. 18 Factory Premises
No. 111 North Dongting Road
Taicang Economy Development Area
215400 Taicang, Jiangsu, P. R. China

CONSIGNEE

TO THE ORDER OF

UNITED BANK LTD., CPU (TRADE),
2ND FLOOR, PRINTING AND STATIONARY BLDG.,
MAI-KOLACHI ROAD, KARACHI, PAKISTAN

NOTIFY PARTY

BULK FLEXIBLE PAKISTAN PVT LTD
501 5TH FLOOR BUSINESS AVENUE
SHAHRAH E FAISAL
KARACHI — PAKISTAN

AND'

PIER OR AIRPORT

IMPORT OF LOADING
SHANGHAI SEAPORT, CHINA

PLACE OF DELIVERY

EXPORTING CARRIER (VESSEL/AIRLINE)

WAN HAI 622 V.W021
AIRISEA PORT OF DISCHARGE

KARACHI SEAPORT, PAKISTAN

Bill of Lading
DOCUMENT NO BOOKING NO
OTHK40020753 WR62B8U70275
EXPORT REFERENCES
MA2053480"""
    }
    
    print("🧪 Testing Improved Document Type Detection")
    print("=" * 60)
    
    for expected_type, content in test_docs.items():
        print(f"\n📄 Testing: {expected_type}")
        print("-" * 40)
        
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
            print("❌ Heuristic detection: INCORRECT")
            
        if full_result == expected_type:
            print("✅ Full detection: CORRECT")
        else:
            print("❌ Full detection: INCORRECT")
            
        print(f"📊 Content length: {len(content)} characters")
        
        # Show first 100 characters for context
        preview = content[:100].replace('\n', ' ').strip()
        print(f"📝 Preview: {preview}...")

if __name__ == "__main__":
    test_document_type_detection()
