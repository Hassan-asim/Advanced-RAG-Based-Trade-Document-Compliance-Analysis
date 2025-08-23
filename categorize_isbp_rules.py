import os
import json
from groq_llm import GROQ_LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import text splitter

# Define paths
temp_isbp_text_file = r'C:\Users\walee\Desktop\1st task\temp_isbp_821_text.txt'
system_prompt_file = r'C:\Users\walee\Desktop\1st task\system_prompt_rule_categorizer.md'
temp_categorized_rules_file = r'C:\Users\walee\Desktop\1st task\temp_categorized_rules.json'

try:
    # Read transcribed ISBP content
    with open(temp_isbp_text_file, 'r', encoding='utf-8') as f:
        isbp_content = f.read()

    # Read system prompt for categorization
    with open(system_prompt_file, 'r', encoding='utf-8') as f:
        system_prompt_content = f.read()

    # Initialize text splitter for LLM input
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000, # Adjust based on model context window and prompt size
        chunk_overlap=500, # Overlap to maintain context across chunks
        length_function=len,
        is_separator_regex=False,
    )
    isbp_chunks = text_splitter.split_text(isbp_content)

    llm = GROQ_LLM()

    # Initialize aggregated results
    aggregated_general_rules = []
    aggregated_document_specific_rules = {
        "BILL OF LADING": [],
        "COMMERCIAL INVOICE": [],
        "COVERING SCHEDULE": [],
        "DHL RECEIPT": [],
        "PACKING LIST": [],
        "SHIPMENT ADVICE": []
    }

    print(f"Sending ISBP-821 content in {len(isbp_chunks)} chunks to LLM for rule categorization...")

    for i, chunk in enumerate(isbp_chunks):
        print(f"  Processing chunk {i+1}/{len(isbp_chunks)}...")
        
        try:
            # The invoke method of GROQ_LLM expects a system_prompt_path and doc_string.
            # It returns a dictionary.
            categorized_chunk_rules = llm.invoke(system_prompt_file, chunk)

            # Aggregate results
            for rule in categorized_chunk_rules.get("general_rules", []):
                if rule not in aggregated_general_rules:
                    aggregated_general_rules.append(rule)

            for doc_type, rules_list in categorized_chunk_rules.get("document_specific_rules", {}).items():
                if doc_type not in aggregated_document_specific_rules:
                    aggregated_document_specific_rules[doc_type] = []
                for rule in rules_list:
                    if rule not in aggregated_document_specific_rules[doc_type]:
                        aggregated_document_specific_rules[doc_type].append(rule)

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Continue processing other chunks even if one fails

    final_categorized_rules = {
        "general_rules": aggregated_general_rules,
        "document_specific_rules": aggregated_document_specific_rules
    }

    # Save final aggregated categorized rules to a temporary JSON file
    with open(temp_categorized_rules_file, 'w', encoding='utf-8') as f:
        json.dump(final_categorized_rules, f, indent=2)
    print(f"LLM categorization complete. Aggregated output saved to {temp_categorized_rules_file}")

except Exception as e:
    print(f"Error during LLM categorization: {e}")
    if os.path.exists(temp_categorized_rules_file):
        os.remove(temp_categorized_rules_file)