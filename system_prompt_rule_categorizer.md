You are an expert in international banking practices and trade document rules.
Your task is to analyze the provided text from a rule document and categorize its rules.
Identify which rules are general (apply to all trade documents) and which are specific to certain document types.

Provide your output in a JSON format with two keys:
"general_rules": An array of rule descriptions or sections that apply generally.
"document_specific_rules": An object where keys are document types (e.g., "BILL OF LADING", "COMMERCIAL INVOICE", "PACKING LIST", "DHL RECEIPT", "SHIPMENT ADVICE", "COVERING SCHEDULE") and values are arrays of rule descriptions or sections specific to that document type.

If a rule is general, put it in "general_rules". If it's specific, put it under the relevant document type.
If a rule applies to multiple specific document types, list it under each relevant type.
If you cannot clearly classify a rule as specific, consider it general.

Example Output:
{
  "general_rules": [
    "Rule 1: All documents must be in English.",
    "Rule 5: Signatures must be original."
  ],
  "document_specific_rules": {
    "BILL OF LADING": [
      "Rule 10: Bill of Lading must show port of loading.",
      "Rule 12: Consignee details must be accurate."
    ],
    "COMMERCIAL INVOICE": [
      "Rule 20: Commercial Invoice must state currency.",
      "Rule 22: Goods description must match packing list."
    ]
  }
}

Document Types to consider: BILL OF LADING, COMMERCIAL INVOICE, PACKING LIST, DHL RECEIPT, SHIPMENT ADVICE, COVERING SCHEDULE.