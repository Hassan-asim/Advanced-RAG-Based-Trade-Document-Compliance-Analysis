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
        # Check for strong indicators first
        if re.search(r"\bbill of lading\b", content_lower):
            return "BILL OF LADING"
        if re.search(r"\bpacking\s+list\b", content_lower):
            return "PACKING LIST"
        if re.search(r"\bcommercial\s+invoice\b", content_lower):
            return "COMMERCIAL INVOICE"
        if re.search(r"\bshipment\s+advice\b", content_lower):
            return "SHIPMENT ADVICE"
        if re.search(r"\bcovering\s+schedule\b", content_lower):
            return "COVERING SCHEDULE"
        if re.search(r"\bdhl\b", content_lower) and re.search(r"\breceipt\b|\bwaybill\b", content_lower):
            return "DHL RECEIPT"
        return None

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_top_k_rules_cached(doc_text: str, rules_text: str, rules_filename: str, k: int) -> tuple:
        # Return as tuple to make it hashable for lru_cache; caller can convert back to list
        rules = get_top_k_rules(
            doc_text=doc_text,
            rule_texts=[rules_text],
            rule_filenames=[rules_filename],
            k=k
        )
        return tuple(rules)

    def get_completion_with_fallback(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024, glm_timeout_seconds: float | None = None) -> str | None:
        print("DEBUG: Attempting completion with GLM (main LLM)...")
        glm_response_content = self.glm_llm.get_completion(messages, temperature=temperature, max_tokens=max_tokens, timeout_seconds=glm_timeout_seconds)
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
                llm_response_content = self.get_completion_with_fallback(messages, max_tokens=max_tokens, glm_timeout_seconds=12)
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

        # Fast heuristic detection
        heuristic = self._heuristic_detect_document_type(document_content)
        if heuristic:
            self._doc_type_cache[doc_hash] = heuristic
            return heuristic

        # Truncate document_content for type detection to avoid token limits
        # Sending only the first 1000 characters as a sample for classification
        truncated_content = document_content[:1000]
        print(f"DEBUG: Truncating document content for type detection. Original length: {len(document_content)}, Truncated length: {len(truncated_content)}")

        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": truncated_content}
        ]

        try:
            # Low token, low latency detection
            llm_response_content = self.get_completion_with_fallback(messages, temperature=0.0, max_tokens=16, glm_timeout_seconds=5)
            if llm_response_content:
                doc_type = llm_response_content.strip().upper()
                self._doc_type_cache[doc_hash] = doc_type
                return doc_type
            else:
                return "UNKNOWN"
        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            return "UNKNOWN"