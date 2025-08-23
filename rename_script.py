import os

old_name = "C:\\Users\\walee\\Desktop\\1st task\\groq_llm.py"
new_name = "C:\\Users\\walee\\Desktop\\1st task\\llm_service.py"

try:
    os.rename(old_name, new_name)
    print(f"Successfully renamed {old_name} to {new_name}")
except FileNotFoundError:
    print(f"Error: File not found at {old_name}")
except Exception as e:
    print(f"Error renaming file: {e}")

