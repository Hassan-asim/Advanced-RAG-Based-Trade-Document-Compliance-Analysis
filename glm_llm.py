import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class GLM_LLM:
    def __init__(self):
        self.api_key = os.environ.get("GLM_API_KEY")
        self.base_url = "https://api.z.ai/api/paas/v4/chat/completions"
        self.headers = {
            'Accept-Language': 'en-US,en;q=0.9', # Default, can be customized
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        self.model = "glm-4.5" # Default model for GLM

    def get_completion(self, messages, temperature=0.7, max_tokens=1024):
        if not self.api_key:
            print("GLM_API_KEY not found in .env. Skipping GLM call.")
            return None

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens # Assuming max_tokens is supported
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for HTTP errors
            response_json = response.json()
            return response_json['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"Error getting completion from GLM: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected response format from GLM: {e}. Response: {response.text}")
            return None