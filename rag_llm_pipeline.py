import os
import json
import re
import functools
import hashlib
from typing import List, Dict
from logging import Logger
from dotenv import load_dotenv
import requests
from groq import Groq
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from vectorizer import get_top_k_rules

# Load environment variables from .env file
load_dotenv()

logger = Logger(__name__)

class GLM_LLM_Client:
    def __init__(self):
        self.api_key = os.environ.get("GLM_API_KEY")
        self.base_url = "https://api.z.ai/api/paas/v4/chat/completions"
        self.headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        self.model = "glm-4.5-flash" # Free model for GLM
        # Reuse a persistent HTTP session to reduce connection overhead
        self.session = requests.Session()
        self.request_timeout_seconds = 12

    def get_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024, timeout_seconds: float | None = None) -> str | None:
        print("DEBUG: Inside GLM_LLM_Client.get_completion")
        if not self.api_key or self.api_key == "your_glm_api_key_here":
            print("DEBUG: GLM_API_KEY not found or not set. Returning None.")
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        print(f"DEBUG: GLM Payload (first 500 chars): {str(payload)[:500]}...")

        response = None
        try:
            response = self.session.post(self.base_url, headers=self.headers, data=json.dumps(payload), timeout=timeout_seconds or self.request_timeout_seconds)
            response.raise_for_status()
            response_json = response.json()
            print(f"DEBUG: GLM Full Response: {json.dumps(response_json, indent=2)}")
            
            content = response_json['choices'][0]['message']['content']
            print(f"DEBUG: GLM Content: '{content}' (length: {len(content) if content else 0})")
            
            if not content or content.strip() == "":
                print("DEBUG: GLM returned empty content")
                return None
                
            return content
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: GLM Request Error: {e}")
            if response is not None:
                print(f"DEBUG: GLM Error Response Text: {response.text}")
            return None
        except KeyError as e:
            print(f"DEBUG: GLM KeyError: {e}. Unexpected response format.")
            if response is not None:
                print(f"DEBUG: GLM Response Text (KeyError): {response.text}")
            return None
        except Exception as e:
            print(f"DEBUG: GLM Unexpected Error: {e}")
            return None

class GROQ_LLM_Client:
    def __init__(self, model_name: str = 'llama3-70b-8192'):
        self.model_name = model_name
        self.temperature = 0.0
        self.max_completion_tokens = 8192
        self.top_p = 1
        self.stop = None
        self.stream = False
        # Reuse a single Groq client instance per process for performance
        groq_api_key = os.environ.get('GROQ_API_KEY')
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None

    def get_completion(self, messages: List[Dict[str, str]], use_json_format: bool = True, max_tokens_override: int | None = None) -> str | None:
        print("DEBUG: Inside GROQ_LLM_Client.get_completion")
        groq_api_key = os.environ.get('GROQ_API_KEY')
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            print("DEBUG: GROQ_API_KEY not found or not set. Returning None.")
            return None

        client = self.client or Groq(api_key=groq_api_key)

        try:
            # Only use JSON format if the message contains 'json' or we're doing compliance analysis
            completion_args = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": max_tokens_override if max_tokens_override is not None else self.max_completion_tokens,
                "top_p": self.top_p,
                "stream": self.stream,
                "stop": self.stop,
            }
            
            # Add JSON format only if appropriate
            if use_json_format:
                # Check if the message content contains 'json' or compliance-related terms
                message_text = str(messages).lower()
                if 'json' in message_text or 'compliance' in message_text or 'report' in message_text:
                    completion_args["response_format"] = {"type": "json_object"}
            
            completion = client.chat.completions.create(**completion_args)
            print(f"DEBUG: Groq Success - returning content")
            return completion.choices[0].message.content
        except Exception as e:
            print(f"DEBUG: Groq Error: {e}")
            return None

class RAGLLMPipeline:
    def __init__(self):
        self.glm_llm = GLM_LLM_Client()
        self.groq_llm = GROQ_LLM_Client()
        # Simple in-memory cache for RAG retrieval keyed by (doc_hash, rules_hash, k)
        # Use an lru_cache-wrapped helper to avoid recomputation across reruns
        self._doc_type_cache: Dict[str, str] = {}

    @staticmethod
    def _heuristic_detect_document_type(document_content: str) -> str | None:
        content_lower = document_content.lower()
        
        # Enhanced DHL RECEIPT detection with OCR error tolerance
        dhl_indicators = [
            r"\bdhl\b",  # Exact DHL
            r"\bd\s*h\s*l\b",  # D H L with spaces
            r"\bd\s*[h]\s*l\b",  # D H L with OCR errors
            r"\bwaybill\b",  # Waybill keyword
            r"\bway\s*bill\b",  # Way bill with space
            r"\btracking\s*number\b",  # Tracking number
            r"\btrack\s*no\b",  # Track no
            r"\bairway\s*bill\b",  # Airway bill
            r"\bair\s*way\s*bill\b",  # Air way bill
            r"\bexpress\s*service\b",  # Express service
            r"\bdelivery\s*receipt\b",  # Delivery receipt
            r"\bdelivery\s*note\b",  # Delivery note
            r"\bparcel\s*receipt\b",  # Parcel receipt
            r"\bparcel\s*note\b",  # Parcel note
            r"\bshipping\s*label\b",  # Shipping label
            r"\bship\s*label\b",  # Ship label
            r"\bconsignment\s*note\b",  # Consignment note
            r"\bconsign\s*note\b",  # Consign note
            r"\bdispatch\s*note\b",  # Dispatch note
            r"\bdispatch\s*advice\b",  # Dispatch advice
        ]
        
        # Check for DHL RECEIPT with multiple indicators
        dhl_score = 0
        for pattern in dhl_indicators:
            if re.search(pattern, content_lower):
                dhl_score += 1
        
        # Enhanced COMMERCIAL INVOICE detection
        invoice_indicators = [
            r"\bcommercial\s*invoice\b",  # Exact match
            r"\binvoice\s*no\b",  # Invoice number
            r"\binvoice\s*date\b",  # Invoice date
            r"\btotal\s*invoice\s*value\b",  # Total invoice value
            r"\btotal\s*value\b",  # Total value
            r"\binvoice\s*value\b",  # Invoice value
            r"\bcfr\b",  # Cost and Freight
            r"\bcif\b",  # Cost, Insurance, and Freight
            r"\bfob\b",  # Free on Board
            r"\bex\s*works\b",  # Ex works
            r"\bcurrency\b",  # Currency mentioned
            r"\bunit\s*price\b",  # Unit price
            r"\bprice\s*per\s*unit\b",  # Price per unit
            r"\brate\b",  # Rate
            r"\bquantity\b",  # Quantity
            r"\bqty\b",  # Qty
            r"\bamount\b",  # Amount
            r"\bnet\s*weight\b",  # Net weight
            r"\bgross\s*weight\b",  # Gross weight
            r"\bpackaging\b",  # Packaging details
            r"\bpayment\s*terms\b",  # Payment terms
            r"\bpayment\s*conditions\b",  # Payment conditions
            r"\bterms\s*of\s*payment\b",  # Terms of payment
            r"\bbuyer\b",  # Buyer
            r"\bseller\b",  # Seller
            r"\bvendor\b",  # Vendor
            r"\bsupplier\b",  # Supplier
            r"\bpurchaser\b",  # Purchaser
            r"\bcustomer\b",  # Customer
            r"\bgoods\s*description\b",  # Goods description
            r"\bdescription\s*of\s*goods\b",  # Description of goods
            r"\bproduct\s*description\b",  # Product description
            r"\bincoterms\b",  # Incoterms
            r"\bexport\s*references\b",  # Export references
            r"\bjob\s*no\b",  # Job number
            r"\bcontract\s*no\b",  # Contract number
            r"\border\s*no\b",  # Order number
            r"\bpurchase\s*order\b",  # Purchase order
            r"\bpo\s*no\b",  # PO number
        ]
        
        invoice_score = 0
        for pattern in invoice_indicators:
            if re.search(pattern, content_lower):
                invoice_score += 1
        
        # Enhanced BILL OF LADING detection
        bol_indicators = [
            r"\bbill\s*of\s*lading\b",  # Exact match
            r"\bocean\s*freight\b",  # Ocean freight
            r"\bshipper\s*exporter\b",  # Shipper/exporter
            r"\bshipper\b",  # Shipper
            r"\bexporter\b",  # Exporter
            r"\bconsignee\b",  # Consignee
            r"\bnotify\s*party\b",  # Notify party
            r"\bnotify\b",  # Notify
            r"\bport\s*of\s*loading\b",  # Port of loading
            r"\bport\s*of\s*discharge\b",  # Port of discharge
            r"\bport\s*of\s*unloading\b",  # Port of unloading
            r"\bvessel\b",  # Vessel
            r"\bship\b",  # Ship
            r"\bcarrier\b",  # Carrier
            r"\bcontainer\s*no\b",  # Container number
            r"\bcontainer\s*number\b",  # Container number
            r"\bseal\s*no\b",  # Seal number
            r"\bseal\s*number\b",  # Seal number
            r"\bshipped\s*on\s*board\b",  # Shipped on board
            r"\bon\s*board\s*date\b",  # On board date
            r"\bnon\s*negotiable\b",  # Non-negotiable
            r"\bocean\s*track\b",  # Ocean track
            r"\bnvocc\b",  # NVOCC
            r"\bforwarding\s*agent\b",  # Forwarding agent
            r"\bfreight\s*forwarder\b",  # Freight forwarder
            r"\btransport\s*company\b",  # Transport company
            r"\bshipping\s*line\b",  # Shipping line
            r"\bocean\s*carrier\b",  # Ocean carrier
            r"\bvoyage\s*no\b",  # Voyage number
            r"\bvoyage\s*number\b",  # Voyage number
            r"\broute\b",  # Route
            r"\bshipping\s*route\b",  # Shipping route
            r"\btransit\s*time\b",  # Transit time
            r"\bdelivery\s*terms\b",  # Delivery terms
            r"\bshipping\s*terms\b",  # Shipping terms
            r"\bfreight\s*terms\b",  # Freight terms
            r"\bfreight\s*prepaid\b",  # Freight prepaid
            r"\bfreight\s*collect\b",  # Freight collect
            r"\bcharter\s*party\b",  # Charter party
            r"\bcharter\s*party\s*bill\b",  # Charter party bill
            r"\bhouse\s*bill\b",  # House bill
            r"\bmaster\s*bill\b",  # Master bill
            r"\bstraight\s*bill\b",  # Straight bill
            r"\border\s*bill\b",  # Order bill
            r"\bnegotiable\s*bill\b",  # Negotiable bill
        ]
        
        bol_score = 0
        for pattern in bol_indicators:
            if re.search(pattern, content_lower):
                bol_score += 1
        
        # Special case: If BOL is mentioned in a list context (like in covering schedules),
        # reduce the score to prevent misclassification
        if "bill of lading" in content_lower or "konnossement" in content_lower:
            # Check if this looks like a list of documents rather than the main document
            if any(term in content_lower for term in ["1st mail", "2nd mail", "mail of documents", "please find enclosed", "documents for"]):
                # This is likely a covering schedule listing BOL as one of the documents
                bol_score = max(0, bol_score - 3)  # Reduce score
                print("DEBUG: Reduced BOL score - appears to be listed in covering schedule context")
        
        # Enhanced PACKING LIST detection
        packing_indicators = [
            r"\bpacking\s*list\b",  # Exact match
            r"\bpackage\s*list\b",  # Package list
            r"\bpackages\b",  # Packages
            r"\bpackage\s*numbers\b",  # Package numbers
            r"\bpackage\s*nos\b",  # Package nos
            r"\bpackaging\s*details\b",  # Packaging details
            r"\bcontents\s*list\b",  # Contents list
            r"\bitem\s*list\b",  # Item list
            r"\bgoods\s*list\b",  # Goods list
            r"\bpacking\s*instructions\b",  # Packing instructions
            r"\bpacking\s*details\b",  # Packing details
            r"\bpackage\s*contents\b",  # Package contents
            r"\bpackage\s*description\b",  # Package description
        ]
        
        packing_score = 0
        for pattern in packing_indicators:
            if re.search(pattern, content_lower):
                packing_score += 1
        
        # Enhanced SHIPMENT ADVICE detection
        shipment_indicators = [
            r"\bshipment\s*advice\b",  # Exact match
            r"\bshipping\s*advice\b",  # Shipping advice
            r"\bshipment\s*notification\b",  # Shipment notification
            r"\bshipping\s*notification\b",  # Shipping notification
            r"\badvice\s*of\s*shipment\b",  # Advice of shipment
            r"\bshipment\s*details\b",  # Shipment details
            r"\bshipping\s*details\b",  # Shipping details
            r"\bshipment\s*information\b",  # Shipment information
            r"\bshipping\s*information\b",  # Shipping information
            r"\bshipment\s*status\b",  # Shipment status
            r"\bshipping\s*status\b",  # Shipping status
        ]
        
        shipment_score = 0
        for pattern in shipment_indicators:
            if re.search(pattern, content_lower):
                shipment_score += 1
        
        # Enhanced COVERING SCHEDULE detection
        covering_indicators = [
            r"\bcovering\s*schedule\b",  # Exact match
            r"\bschedule\s*of\s*documents\b",  # Schedule of documents
            r"\bdocument\s*schedule\b",  # Document schedule
            r"\battachments\s*list\b",  # Attachments list
            r"\bsupporting\s*documents\b",  # Supporting documents
            r"\bdocument\s*list\b",  # Document list
            r"\battachments\b",  # Attachments
            r"\bsupporting\s*docs\b",  # Supporting docs
            r"\bdocument\s*attachments\b",  # Document attachments
            r"\bschedule\s*of\s*attachments\b",  # Schedule of attachments
        ]
        
        covering_score = 0
        for pattern in covering_indicators:
            if re.search(pattern, content_lower):
                covering_score += 1
        
        # Document structure analysis to improve accuracy
        # Look for document headers and titles throughout the document
        lines = document_content.split('\n')
        first_lines = [line.strip().lower() for line in lines[:15] if line.strip()]  # First 15 lines
        
        # Also search for document type indicators throughout the entire document
        all_lines = [line.strip().lower() for line in lines if line.strip()]
        
        # Check for document type in first few lines (more reliable)
        header_boost = 0
        for line in first_lines:
            if "packing list" in line:
                packing_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found PACKING LIST in header: '{line}'")
            elif "shipment advice" in line:
                shipment_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found SHIPMENT ADVICE in header: '{line}'")
            elif "covering schedule" in line:
                covering_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found COVERING SCHEDULE in header: '{line}'")
            elif "commercial invoice" in line:
                invoice_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found COMMERCIAL INVOICE in header: '{line}'")
            elif "bill of lading" in line:
                bol_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found BILL OF LADING in header: '{line}'")
            elif "dhl" in line or "waybill" in line:
                dhl_score += 5  # Strong boost for header match
                header_boost += 1
                print(f"DEBUG: Found DHL/WAYBILL in header: '{line}'")
        
        # Search for document type indicators throughout the entire document
        for line in all_lines:
            if "shipment advice" in line and "shipment advice" not in str(first_lines):
                shipment_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found SHIPMENT ADVICE in document: '{line}'")
            elif "covering schedule" in line and "covering schedule" not in str(first_lines):
                covering_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found COVERING SCHEDULE in document: '{line}'")
            elif "packing list" in line and "packing list" not in str(first_lines):
                packing_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found PACKING LIST in document: '{line}'")
            elif "commercial invoice" in line and "commercial invoice" not in str(first_lines):
                invoice_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found COMMERCIAL INVOICE in document: '{line}'")
            elif "bill of lading" in line and "bill of lading" not in str(first_lines):
                bol_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found BILL OF LADING in document: '{line}'")
            elif ("dhl" in line or "waybill" in line) and "dhl" not in str(first_lines) and "waybill" not in str(first_lines):
                dhl_score += 5  # Strong boost for document type found anywhere
                print(f"DEBUG: Found DHL/WAYBILL in document: '{line}'")
        
        # Special handling for COVERING SCHEDULE - it's a meta-document that lists other documents
        if "covering schedule" in document_content.lower() or "schedule of documents" in document_content.lower():
            covering_score += 10  # Very strong boost
            print("DEBUG: Strong boost for COVERING SCHEDULE based on content")
        
        # Additional COVERING SCHEDULE detection - look for meta-document patterns
        # Only apply this boost if we have strong evidence it's a covering schedule
        covering_indicators_found = 0
        if any(term in document_content.lower() for term in [
            "please find enclosed the following documents",
            "enclosed the following documents",
            "documents for",
            "1st mail", "2nd mail",
            "draft", "konnossement",
            "mail of documents",
            "documentary credit",
            "our reference date",
            "your reference"
        ]):
            covering_indicators_found += 1
        
        # Special strong boost for "mail of documents" pattern
        if "mail of documents" in document_content.lower():
            covering_score += 8  # Strong boost for this specific pattern
            covering_indicators_found += 1
            print("DEBUG: Strong boost for 'mail of documents' pattern")
        
        # Additional strong indicators for COVERING SCHEDULE
        if any(term in document_content.lower() for term in [
            "covering schedule",
            "schedule of documents",
            "document schedule",
            "attachments list",
            "supporting documents",
            "document list",
            "attachments",
            "supporting docs",
            "document attachments",
            "schedule of attachments"
        ]):
            covering_indicators_found += 1
        
        # Special case: If document references multiple document types, it's likely a COVERING SCHEDULE
        # Count how many different document types are referenced
        document_type_references = 0
        if "commercial invoice" in document_content.lower():
            document_type_references += 1
        if "packing list" in document_content.lower():
            document_type_references += 1
        if "shipping advice" in document_content.lower() or "shipment advice" in document_content.lower():
            document_type_references += 1
        if "bill of lading" in document_content.lower() or "konnossement" in document_content.lower():
            document_type_references += 1
        if "draft" in document_content.lower():
            document_type_references += 1
        
        # If document references multiple document types, it's likely a covering schedule
        if document_type_references >= 3:
            covering_indicators_found += 3  # Very strong boost
            print(f"DEBUG: Very strong boost for COVERING SCHEDULE - references {document_type_references} document types")
        elif document_type_references >= 2:
            covering_indicators_found += 2  # Strong boost
            print(f"DEBUG: Strong boost for COVERING SCHEDULE - references {document_type_references} document types")
        
        # Only give the boost if we have multiple strong indicators
        if covering_indicators_found >= 3:
            covering_score += 15  # Very strong boost for covering schedule
            print("DEBUG: Very strong boost for COVERING SCHEDULE based on multiple indicators")
        elif covering_indicators_found >= 2:
            covering_score += 10  # Strong boost for covering schedule
            print("DEBUG: Strong boost for COVERING SCHEDULE based on multiple indicators")
        elif covering_indicators_found == 1:
            covering_score += 5  # Moderate boost for single indicator
            print("DEBUG: Moderate boost for COVERING SCHEDULE based on single indicator")
        
        # Special handling for PACKING LIST - look for specific packaging indicators
        if "packaging:" in document_content.lower() or "package nos" in document_content.lower():
            packing_score += 3
            print("DEBUG: Boost for PACKING LIST based on packaging indicators")
        
        # Special handling for SHIPMENT ADVICE - look for shipment-specific content
        if "shipment advice" in document_content.lower() or "shipping advice" in document_content.lower():
            shipment_score += 3
            print("DEBUG: Boost for SHIPMENT ADVICE based on content")
        
        # Special handling for SHIPMENT ADVICE - look for shipment-specific indicators
        if any(term in document_content.lower() for term in ["shipment details", "shipping details", "vessel name", "shipped on board date", "expected arrival date"]):
            shipment_score += 2
            print("DEBUG: Boost for SHIPMENT ADVICE based on shipment indicators")
        
        # Penalize documents that have too many mixed indicators
        # This helps distinguish between primary and secondary content
        total_indicators = dhl_score + invoice_score + bol_score + packing_score + shipment_score + covering_score
        
        # If a document has too many mixed indicators, reduce scores for less specific types
        if total_indicators > 20:  # Increased threshold
            # Reduce scores for types that commonly overlap
            if invoice_score > 0 and bol_score > 0:
                # If both invoice and BOL indicators exist, reduce the lower score
                if invoice_score < bol_score:
                    invoice_score = max(0, invoice_score - 3)
                    print(f"DEBUG: Reduced invoice score due to overlap with BOL")
                else:
                    bol_score = max(0, bol_score - 3)
                    print(f"DEBUG: Reduced BOL score due to overlap with invoice")
            
            # Reduce invoice score if document is clearly not an invoice
            if invoice_score > 0 and (packing_score > 0 or shipment_score > 0 or covering_score > 0):
                if "packing list" in document_content.lower() or "shipment advice" in document_content.lower() or "covering schedule" in document_content.lower():
                    invoice_score = max(0, invoice_score - 2)
                    print(f"DEBUG: Reduced invoice score for non-invoice document type")
        
        # Special penalty for BOL when we have strong COVERING SCHEDULE evidence
        # This prevents COVERING SCHEDULE from being misclassified as BOL
        if covering_indicators_found >= 2 and bol_score > 0:
            # If we have strong evidence it's a covering schedule, heavily penalize BOL
            # because covering schedules often list BOL documents but aren't BOLs themselves
            if "mail of documents" in document_content.lower() or "please find enclosed" in document_content.lower():
                bol_score = max(0, bol_score - 15)  # Much stronger penalty
                print("DEBUG: Very strong penalty for BOL due to strong covering schedule evidence")
            elif covering_indicators_found >= 3:
                bol_score = max(0, bol_score - 12)  # Stronger penalty
                print("DEBUG: Strong penalty for BOL due to strong covering schedule evidence")
            else:
                bol_score = max(0, bol_score - 8)  # Stronger penalty
                print("DEBUG: Moderate penalty for BOL due to covering schedule evidence")
        
        # Additional penalty: If this is clearly a covering schedule (multiple strong indicators),
        # heavily penalize BOL to prevent misclassification
        if covering_indicators_found >= 3 and "mail of documents" in document_content.lower():
            # This is almost certainly a covering schedule, so heavily penalize BOL
            bol_score = max(0, bol_score - 20)  # Very heavy penalty
            print("DEBUG: Very heavy penalty for BOL - document is clearly a covering schedule")
        
        # Special case: If document contains multiple document types listed, it's likely a COVERING SCHEDULE
        # But only if we have strong evidence (multiple covering indicators)
        if covering_score > 0 and covering_indicators_found >= 2 and (invoice_score > 0 or bol_score > 0 or packing_score > 0 or shipment_score > 0):
            # Boost covering schedule for documents that list other document types
            covering_score += 8  # Increased boost
            print("DEBUG: Strong boost for COVERING SCHEDULE due to multiple document types listed")
            
            # Penalize other document types when we have strong covering schedule evidence
            if covering_indicators_found >= 3:
                # Reduce scores for other types to prevent misclassification
                if invoice_score > 0:
                    invoice_score = max(0, invoice_score - 5)  # Increased penalty
                    print("DEBUG: Reduced invoice score due to strong covering schedule evidence")
                if bol_score > 0:
                    bol_score = max(0, bol_score - 8)  # Much stronger penalty for BOL
                    print("DEBUG: Reduced BOL score due to strong covering schedule evidence")
                if packing_score > 0:
                    packing_score = max(0, packing_score - 5)  # Increased penalty
                    print("DEBUG: Reduced packing score due to strong covering schedule evidence")
                if shipment_score > 0:
                    shipment_score = max(0, shipment_score - 5)  # Increased penalty
                    print("DEBUG: Reduced shipment score due to strong covering schedule evidence")
                if dhl_score > 0:
                    dhl_score = max(0, dhl_score - 5)  # Increased penalty
                    print("DEBUG: Reduced DHL score due to strong covering schedule evidence")
            elif covering_indicators_found >= 2:
                # Moderate penalties for moderate evidence
                if bol_score > 0:
                    bol_score = max(0, bol_score - 4)  # Moderate penalty for BOL
                    print("DEBUG: Reduced BOL score due to moderate covering schedule evidence")
                if invoice_score > 0:
                    invoice_score = max(0, invoice_score - 2)
                    print("DEBUG: Reduced invoice score due to moderate covering schedule evidence")
        
        # Score-based classification with confidence thresholds
        scores = {
            "DHL RECEIPT": dhl_score,
            "COMMERCIAL INVOICE": invoice_score,
            "BILL OF LADING": bol_score,
            "PACKING LIST": packing_score,
            "SHIPMENT ADVICE": shipment_score,
            "COVERING SCHEDULE": covering_score
        }
        
        # Find the document type with the highest score
        max_score = max(scores.values())
        max_score_types = [doc_type for doc_type, score in scores.items() if score == max_score]
        
        print(f"DEBUG: Final scores: {scores}")
        print(f"DEBUG: Header boost: {header_boost}")
        
        # Only return a result if we have a clear winner with sufficient confidence
        if max_score >= 5 and len(max_score_types) == 1:
            detected_type = max_score_types[0]
            print(f"DEBUG: Heuristic detection successful - {detected_type} (score: {max_score})")
            return detected_type
        elif max_score >= 3 and len(max_score_types) == 1:
            detected_type = max_score_types[0]
            print(f"DEBUG: Heuristic detection with medium confidence - {detected_type} (score: {max_score})")
            return detected_type
        elif max_score >= 2 and len(max_score_types) == 1 and header_boost > 0:
            detected_type = max_score_types[0]
            print(f"DEBUG: Heuristic detection with header boost - {detected_type} (score: {max_score}, header_boost: {header_boost})")
            return detected_type
        
        print(f"DEBUG: Heuristic detection failed - scores: {scores}, header_boost: {header_boost}")
        return None

    @staticmethod
    def _get_top_k_rules_cached(doc_text: str, rules_text: str, rules_filename: str, k: int) -> tuple:
        # Direct call without caching to ensure fresh analysis results
        rules = get_top_k_rules(
            doc_text=doc_text,
            rule_texts=[rules_text],
            rule_filenames=[rules_filename],
            k=k
        )
        return tuple(rules)

    def get_completion_with_fallback(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024, glm_timeout_seconds: float | None = None) -> str | None:
        print("DEBUG: Attempting completion with GLM (main LLM)...")
        # Reduce GLM timeout to failover faster during analysis
        glm_response_content = self.glm_llm.get_completion(messages, temperature=temperature, max_tokens=max_tokens, timeout_seconds=glm_timeout_seconds or 6)
        if glm_response_content and glm_response_content.strip():
            print("DEBUG: GLM returned content.")
            return glm_response_content
        
        print("DEBUG: GLM failed or returned empty. Falling back to Groq LLM...")
        # Determine if we need JSON format based on message content
        message_text = str(messages).lower()
        use_json = 'json' in message_text or 'compliance' in message_text or 'report' in message_text
        
        groq_response_content = self.groq_llm.get_completion(messages, use_json_format=use_json, max_tokens_override=max_tokens)
        if groq_response_content and groq_response_content.strip():
            print("DEBUG: Groq returned content.")
            return groq_response_content
        
        print("DEBUG: Both GLM and Groq failed.")
        return None

    def process_document_for_compliance(self, document: Dict[str, str], rules_text: str, rules_filename: str) -> Dict:
        """
        Process a document for compliance analysis using RAG approach.
        Instead of sending the entire rules_text, use vectorized retrieval to get relevant chunks.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(base_path, 'system_prompt.md')
        with open(system_prompt_path, encoding="utf-8") as file:
            system_prompt_content = file.read()

        doc_content = f"""--- DOCUMENT TO ANALYZE: {document['filename']} ---
{document['content']}"""

        print(f"DEBUG: Original rules text length: {len(rules_text)} characters")
        print("DEBUG: Using RAG to retrieve relevant rule chunks...")
        
        # Use memoized RAG retrieval for top-k most relevant rule chunks
        relevant_rules = list(self._get_top_k_rules_cached(
            document['content'],
            rules_text,
            rules_filename,
            10
        ))
        
        # Deduplicate while preserving order
        seen = set()
        deduped_rules = []
        for chunk in relevant_rules:
            key = chunk.strip()
            if key not in seen:
                seen.add(key)
                deduped_rules.append(chunk)
        
        # Combine the relevant rules into a smaller context and normalize whitespace
        relevant_rules_text = "\n\n".join(deduped_rules)
        # Normalize excessive whitespace to reduce token count without changing semantics
        relevant_rules_text = re.sub(r"\s+", " ", relevant_rules_text).replace(" \n ", "\n").strip()
        print(f"DEBUG: Relevant rules text length after RAG: {len(relevant_rules_text)} characters")
        
        # Ensure we don't exceed reasonable token limits (roughly 4000 tokens = ~16k chars)
        if len(relevant_rules_text) > 15000:
            print("DEBUG: Still too long, taking top 5 chunks only (no re-vectorize)")
            relevant_rules = list(deduped_rules[:5])
            relevant_rules_text = "\n\n".join(relevant_rules)
            relevant_rules_text = re.sub(r"\s+", " ", relevant_rules_text).replace(" \n ", "\n").strip()
            print(f"DEBUG: Final relevant rules text length: {len(relevant_rules_text)} characters")

        def build_messages(rules_txt: str) -> List[Dict[str, str]]:
            combined_prompt = (
                f"<RULES_TEXT FILENAME='{rules_filename}'>\n{rules_txt}\n</RULES_TEXT>\n\n"
                f"<USER_DOCUMENT>\n{doc_content}\n</USER_DOCUMENT>"
            )
            return [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": combined_prompt}
            ]

        messages = build_messages(relevant_rules_text)

        print(f"DEBUG: Total message length: {len(str(messages))} characters")
        print(f"DEBUG: Document content length: {len(document['content'])} characters")
        print(f"DEBUG: Final rules text length: {len(relevant_rules_text)} characters")

        structured_response = {}
        max_tokens = 2048  # Starting token limit

        # Parallel shard analysis if many relevant chunks; preserve behavior by merging results
        shard_threshold = 8
        max_shards = 4
        if len(deduped_rules) >= shard_threshold:
            shard_count = min(max_shards, math.ceil(len(deduped_rules) / 3))
            shard_size = math.ceil(len(deduped_rules) / shard_count)
            rule_shards = [deduped_rules[i:i+shard_size] for i in range(0, len(deduped_rules), shard_size)]

            def analyze_shard(rules_subset: List[str]) -> Dict:
                shard_rules_text = "\n\n".join(rules_subset)
                shard_rules_text = re.sub(r"\s+", " ", shard_rules_text).replace(" \n ", "\n").strip()
                shard_messages = build_messages(shard_rules_text)
                # Use a slightly lower max tokens per shard; fallback retains JSON parsing logic below
                return self._analyze_messages_with_retry(shard_messages, document, initial_max_tokens=1536)

            merged_discrepancies = []
            merged_compliances = []
            try:
                with ThreadPoolExecutor(max_workers=shard_count) as executor:
                    futures = {executor.submit(analyze_shard, shard): shard for shard in rule_shards}
                    for fut in as_completed(futures):
                        shard_result = fut.result()
                        if isinstance(shard_result, dict):
                            report_list = shard_result.get("compliance_report", [])
                            if report_list:
                                report = report_list[0]
                                merged_discrepancies.extend(report.get("discrepancies", []))
                                merged_compliances.extend(report.get("compliances", []))

                # Dedupe by (finding, rule)
                def dedupe(items: List[Dict]) -> List[Dict]:
                    seen_keys = set()
                    unique_items = []
                    for item in items:
                        key = (item.get('finding', '').strip(), item.get('rule', '').strip())
                        if key not in seen_keys:
                            seen_keys.add(key)
                            unique_items.append(item)
                    return unique_items

                merged_discrepancies = dedupe(merged_discrepancies)
                merged_compliances = dedupe(merged_compliances)

                # Deterministic ordering for consistency
                merged_discrepancies.sort(key=lambda x: (x.get('rule', ''), x.get('finding', '')))
                merged_compliances.sort(key=lambda x: (x.get('rule', ''), x.get('finding', '')))

                structured_response = {
                    "compliance_report": [{
                        "document_name": document.get('filename', 'unknown'),
                        "discrepancies": merged_discrepancies,
                        "compliances": merged_compliances
                    }]
                }
                return structured_response
            except Exception:
                # Fallback to single-call path below
                pass
        
        for attempt in range(2):  # Try twice with different token limits
            try:
                print(f"DEBUG: Attempt {attempt + 1} with max_tokens={max_tokens}")
                # Use higher max_tokens for compliance analysis (needs more detailed output)
                llm_response_content = self.get_completion_with_fallback(messages, temperature=0.0, max_tokens=max_tokens, glm_timeout_seconds=12)
                if llm_response_content:
                    # Clean the response in case it has markdown formatting
                    clean_response = llm_response_content.strip()
                    
                    # Remove markdown code blocks
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    elif clean_response.startswith("```"):
                        clean_response = clean_response[3:]
                        
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                        
                    clean_response = clean_response.strip()
                    
                    # Handle truncated JSON - try to find complete JSON objects
                    is_truncated = not clean_response.endswith('}') and not clean_response.endswith(']')
                    
                    if is_truncated:
                        print("DEBUG: Response appears truncated, attempting to find complete JSON")
                        # Find the last complete JSON object
                        brace_count = 0
                        last_valid_pos = -1
                        
                        for i, char in enumerate(clean_response):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    last_valid_pos = i + 1
                        
                        if last_valid_pos > 0:
                            clean_response = clean_response[:last_valid_pos]
                            print(f"DEBUG: Truncated response to valid JSON ending at position {last_valid_pos}")
                        
                        # If this is the first attempt and response was truncated, try again with more tokens
                        if attempt == 0:
                            max_tokens = 4096  # Increase for next attempt
                            print("DEBUG: Response was truncated, will retry with more tokens")
                            continue
                    
                    # Try to parse the cleaned response
                    try:
                        structured_response = json.loads(clean_response)
                        print("DEBUG: Successfully parsed JSON response")
                        break  # Success, exit retry loop
                        
                    except json.JSONDecodeError as json_err:
                        # If parsing fails, try to fix common JSON issues
                        print(f"DEBUG: JSON parsing failed, attempting to fix: {json_err}")
                        
                        # Remove any trailing commas and incomplete quotes
                        lines = clean_response.split('\n')
                        fixed_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            # Skip empty lines
                            if not line:
                                continue
                            # Fix incomplete strings (remove unterminated quotes)
                            if line.count('"') % 2 != 0 and not line.endswith('",') and not line.endswith('"'):
                                # Find the last quote and truncate there
                                last_quote = line.rfind('"')
                                if last_quote > 0:
                                    line = line[:last_quote + 1]
                            # Remove trailing commas before closing braces/brackets
                            if line.endswith(',') and (lines.index(line) == len(lines) - 1 or 
                                                       any(next_line.strip().startswith(c) for c in ['}', ']'] 
                                                           for next_line in lines[lines.index(line) + 1:])):
                                line = line[:-1]
                            fixed_lines.append(line)
                        
                        fixed_response = '\n'.join(fixed_lines)
                        
                        # Ensure proper JSON structure
                        if not fixed_response.strip().endswith('}'):
                            # Add missing closing braces
                            open_braces = fixed_response.count('{') - fixed_response.count('}')
                            fixed_response += '}' * open_braces
                        
                        try:
                            structured_response = json.loads(fixed_response)
                            print("DEBUG: Successfully parsed fixed JSON response")
                            break  # Success, exit retry loop
                            
                        except json.JSONDecodeError:
                            # If this is the first attempt, try again with more tokens
                            if attempt == 0:
                                max_tokens = 4096
                                print("DEBUG: Could not fix JSON, will retry with more tokens")
                                continue
                            else:
                                # Final fallback - create minimal valid response
                                print("DEBUG: Could not fix JSON, creating minimal response")
                                structured_response = {
                                    "error": "LLM response was not valid JSON.",
                                    "details": f"JSON parsing failed: {json_err}",
                                    "raw_response": llm_response_content,
                                    "compliance_report": [{
                                        "document_name": document.get('filename', 'unknown'),
                                        "discrepancies": [],
                                        "compliances": []
                                    }]
                                }
                                break
                else:
                    structured_response = {"error": "Both GLM and Groq LLMs failed to provide a response.", "details": "No LLM response content."}
                    break
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e}. Raw response: {llm_response_content}")
                structured_response = {
                    "error": "LLM response was not valid JSON.", 
                    "details": str(e), 
                    "raw_response": llm_response_content,
                    "compliance_report": [{
                        "document_name": document.get('filename', 'unknown'),
                        "discrepancies": [],
                        "compliances": []
                    }]
                }
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred during LLM invocation or JSON parsing: {e}")
                structured_response = {"error": "An unexpected error occurred during LLM invocation.", "details": str(e)}
                break
                
        return structured_response

    def _analyze_messages_with_retry(self, messages: List[Dict[str, str]], document: Dict[str, str], initial_max_tokens: int = 2048) -> Dict:
        max_tokens = initial_max_tokens
        for attempt in range(2):
            try:
                llm_response_content = self.get_completion_with_fallback(messages, max_tokens=max_tokens, glm_timeout_seconds=12)
                if llm_response_content:
                    clean_response = llm_response_content.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    elif clean_response.startswith("```"):
                        clean_response = clean_response[3:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    is_truncated = not clean_response.endswith('}') and not clean_response.endswith(']')
                    if is_truncated:
                        brace_count = 0
                        last_valid_pos = -1
                        for i, char in enumerate(clean_response):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    last_valid_pos = i + 1
                        if last_valid_pos > 0:
                            clean_response = clean_response[:last_valid_pos]
                        if attempt == 0:
                            max_tokens = initial_max_tokens * 2
                            continue
                    try:
                        return json.loads(clean_response)
                    except json.JSONDecodeError:
                        # Try simple fixes
                        lines = clean_response.split('\n')
                        fixed_lines = []
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            if line.count('"') % 2 != 0 and not line.endswith('",') and not line.endswith('"'):
                                last_quote = line.rfind('"')
                                if last_quote > 0:
                                    line = line[:last_quote + 1]
                            if line.endswith(','):
                                line = line[:-1]
                            fixed_lines.append(line)
                        fixed_response = '\n'.join(fixed_lines)
                        if not fixed_response.strip().endswith('}'):
                            open_braces = fixed_response.count('{') - fixed_response.count('}')
                            fixed_response += '}' * max(0, open_braces)
                        try:
                            return json.loads(fixed_response)
                        except json.JSONDecodeError:
                            if attempt == 0:
                                max_tokens = initial_max_tokens * 2
                                continue
                            else:
                                return {
                                    "error": "LLM response was not valid JSON.",
                                    "details": "Parsing failed after retries",
                                    "compliance_report": [{
                                        "document_name": document.get('filename', 'unknown'),
                                        "discrepancies": [],
                                        "compliances": []
                                    }]
                                }
                else:
                    if attempt == 0:
                        max_tokens = initial_max_tokens * 2
                        continue
                    return {"error": "No LLM response.", "compliance_report": [{"document_name": document.get('filename', 'unknown'), "discrepancies": [], "compliances": []}]}
            except Exception:
                if attempt == 0:
                    max_tokens = initial_max_tokens * 2
                    continue
                return {"error": "Exception during analysis.", "compliance_report": [{"document_name": document.get('filename', 'unknown'), "discrepancies": [], "compliances": []}]}

    def detect_document_type(self, document_content: str) -> str:
        base_path = os.path.dirname(os.path.abspath(__file__))
        system_prompt_path = os.path.join(base_path, 'system_prompt_doc_type.md')

        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_content = f.read()

        # Ensure document_content is a string
        if not isinstance(document_content, str):
            print(f"DEBUG: document_content is not a string, type: {type(document_content)}")
            return {"error": f"Expected string content, got {type(document_content)}"}

        # Check in-memory cache first
        doc_hash = hashlib.sha256(document_content.encode('utf-8')).hexdigest()
        cached = self._doc_type_cache.get(doc_hash)
        if cached:
            return cached

        # Fast heuristic detection with enhanced patterns
        heuristic = self._heuristic_detect_document_type(document_content)
        if heuristic:
            self._doc_type_cache[doc_hash] = heuristic
            return heuristic

        # Enhanced LLM-based detection with better sampling
        # Use multiple samples from different parts of the document for better accuracy
        doc_length = len(document_content)
        
        # Create multiple samples for better detection
        samples = []
        
        # Sample 1: First 1500 characters (header/beginning)
        if doc_length > 1500:
            samples.append(f"HEADER SAMPLE:\n{document_content[:1500]}")
        
        # Sample 2: Middle section (if document is long enough)
        if doc_length > 3000:
            middle_start = doc_length // 2 - 750
            middle_end = doc_length // 2 + 750
            samples.append(f"MIDDLE SAMPLE:\n{document_content[middle_start:middle_end]}")
        
        # Sample 3: Last 1500 characters (footer/end)
        if doc_length > 1500:
            samples.append(f"FOOTER SAMPLE:\n{document_content[-1500:]}")
        
        # If document is short, just use the whole thing
        if not samples:
            samples.append(f"FULL DOCUMENT:\n{document_content}")
        
        # Try each sample until we get a confident result
        for i, sample in enumerate(samples):
            print(f"DEBUG: Trying sample {i+1} for document type detection (length: {len(sample)})")
            
            messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": sample}
            ]

            try:
                # Low token, low latency detection with retry
                llm_response_content = self.get_completion_with_fallback(
                    messages, 
                    temperature=0.0, 
                    max_tokens=32,  # Increased for better responses
                    glm_timeout_seconds=8  # Increased timeout
                )
                
                if llm_response_content:
                    doc_type = llm_response_content.strip().upper()
                    
                    # Validate the response is one of our expected types
                    valid_types = {
                        "BILL OF LADING", "COMMERCIAL INVOICE", "PACKING LIST", 
                        "DHL RECEIPT", "SHIPMENT ADVICE", "COVERING SCHEDULE"
                    }
                    
                    if doc_type in valid_types:
                        print(f"DEBUG: LLM detection successful with sample {i+1}: {doc_type}")
                        self._doc_type_cache[doc_hash] = doc_type
                        return doc_type
                    elif doc_type != "UNKNOWN":
                        print(f"DEBUG: LLM returned unexpected type: {doc_type}, trying next sample")
                        continue
                    else:
                        print(f"DEBUG: LLM returned UNKNOWN for sample {i+1}, trying next sample")
                        continue
                else:
                    print(f"DEBUG: No LLM response for sample {i+1}, trying next sample")
                    continue
                    
            except Exception as e:
                print(f"DEBUG: Error with sample {i+1}: {e}, trying next sample")
                continue
        
        # If all samples failed, try one more time with the full document (truncated)
        print("DEBUG: All samples failed, trying with truncated full document")
        try:
            # Use a larger sample but still within reasonable limits
            truncated_content = document_content[:2000] if doc_length > 2000 else document_content
            
            messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": f"FULL DOCUMENT SAMPLE:\n{truncated_content}"}
            ]
            
            llm_response_content = self.get_completion_with_fallback(
                messages, 
                temperature=0.0, 
                max_tokens=32,
                glm_timeout_seconds=10
            )
            
            if llm_response_content:
                doc_type = llm_response_content.strip().upper()
                valid_types = {
                    "BILL OF LADING", "COMMERCIAL INVOICE", "PACKING LIST", 
                    "DHL RECEIPT", "SHIPMENT ADVICE", "COVERING SCHEDULE"
                }
                
                if doc_type in valid_types:
                    print(f"DEBUG: Final LLM detection successful: {doc_type}")
                    self._doc_type_cache[doc_hash] = doc_type
                    return doc_type
        
        except Exception as e:
            print(f"DEBUG: Final detection attempt failed: {e}")
        
        print("DEBUG: All detection methods failed, returning UNKNOWN")
        return "UNKNOWN"