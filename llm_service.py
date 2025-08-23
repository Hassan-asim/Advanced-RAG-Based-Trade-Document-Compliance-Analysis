import os
import json
from uuid import uuid1
from typing import List, Dict
from logging import Logger
from dotenv import load_dotenv
import copy
from groq import Groq
import time

# Import the new GLM LLM
from glm_llm import GLM_LLM

# Load environment variables from .env file
load_dotenv()

logger = Logger(__name__)

class GROQ_LLM_Client(): # Renamed to avoid conflict and clarify role
    def __init__(self, model_name='llama3-70b-8192'):
        self.model_name = model_name
        self.temperature = 0.0
        self.max_completion_tokens = 8192
        self.top_p = 1
        self.stop = None
        self.response_format = {"type": "json_object"}
        self.stream = False

    def get_completion(self, messages):
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stream=self.stream,
                response_format=self.response_format,
                stop=self.stop,
            )
            return completion.choices[0].message.content # Return content directly
        except Exception as e:
            print(f"Error getting completion from Groq: {e}")
            return None

class LLMService:
    def __init__(self):
        self.glm_llm = GLM_LLM()
        self.groq_llm = GROQ_LLM_Client()

    def get_completion_with_fallback(self, messages, use_json_format=True):
        # Try GLM first
        print("Attempting completion with GLM (main LLM)...")
        # GLM_LLM.get_completion expects messages directly
        glm_response_content = self.glm_llm.get_completion(messages)
        if glm_response_content:
            return glm_response_content
        
        print("GLM failed or returned empty. Falling back to Groq LLM...")
        # Groq LLM Client's get_completion now also returns content directly
        groq_response_content = self.groq_llm.get_completion(messages)
        if groq_response_content:
            return groq_response_content
        
        print("Both GLM and Groq failed.")
        return None

# The process_document function needs to be updated to use LLMService
def process_document(document: Dict[str, str], rules_text: str, rules_filename: str, llm_service: LLMService):
    # The system prompt for process_document is read from system_prompt.md
    # This system prompt asks for JSON output.
    base_path = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(base_path, 'system_prompt.md')
    with open(system_prompt_path, encoding="utf-8") as file:
        system_prompt_content = file.read()

        doc_content = f"""--- DOCUMENT TO ANALYZE: {document['filename']} ---"""

    # Combine the rules and the user documents into a single prompt string.
    combined_prompt_string = (
        f"<RULES_TEXT FILENAME='{rules_filename}'>\n{rules_text}\n</RULES_TEXT>\n\n"
        f"<USER_DOCUMENT>\n{doc_content}\n</USER_DOCUMENT>"
    )

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": combined_prompt_string}
    ]

    structured_response = {}
    try:
        # Use the fallback mechanism
        llm_response_content = llm_service.get_completion_with_fallback(messages)
        if llm_response_content:
            structured_response = json.loads(llm_response_content)
        else:
            structured_response = {"error": "Both GLM and Groq LLMs failed to provide a response.", "details": "No LLM response content."}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}. Raw response: {llm_response_content}")
        structured_response = {"error": "LLM response was not valid JSON.", "details": str(e), "raw_response": llm_response_content}
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM invocation or JSON parsing: {e}")
        structured_response = {"error": "An unexpected error occurred during LLM invocation.", "details": str(e)}
            
    return structured_response

# The main() function from the original groq_llm.py is removed as app.py orchestrates.