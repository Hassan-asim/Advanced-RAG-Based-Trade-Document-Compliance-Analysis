# Trade Document Compliance Checker

This project implements a sophisticated Trade Document Compliance Checker, leveraging a Retrieval-Augmented Generation (RAG) pipeline to validate trade documents against international banking rules such as ISBP and UCP. The system is designed to provide a detailed compliance report, highlighting both discrepancies and compliances based on the relevant articles and rules.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Why RAG over Simple Prompt Engineering?](#why-rag-over-simple-prompt-engineering)
- [System Architecture](#system-architecture)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)

## Introduction
In international trade finance, ensuring strict compliance of documents with established rules like the International Standard Banking Practice (ISBP) and Uniform Customs and Practice for Documentary Credits (UCP) is paramount. Manual checking is prone to errors and time-consuming. This application automates the compliance verification process by utilizing advanced AI models to analyze trade documents against a comprehensive knowledge base of rules.

## Features
- **Automated Compliance Checking:** Automatically validates uploaded trade documents (e.g., Bill of Lading, Commercial Invoice, Packing List) against ISBP and UCP rules.
- **Detailed Compliance Reports:** Generates a structured JSON report outlining specific discrepancies (violated rules) and compliances (followed rules).
- **Rule-Specific Validation:** Validates documents against each rule document (e.g., ISBP 745, ISBP 821, UCP 600) independently, providing separate reports for clarity.
- **Retrieval-Augmented Generation (RAG):** Employs a RAG pipeline to efficiently retrieve the most relevant rule sections for analysis, optimizing LLM performance and reducing token usage.
- **User-Friendly Interface:** A Streamlit-based web interface for easy document upload and report viewing.

## Why RAG over Simple Prompt Engineering?

Initially, a simpler prompt engineering approach was considered where the entire rulebook (or multiple rulebooks concatenated) would be fed directly to the Large Language Model (LLM) along with the document to be analyzed. While this might work for very small rule sets or documents, it quickly becomes unfeasible and inefficient for several critical reasons:

1.  **Token Limit Constraints:** LLMs have strict input token limits. Comprehensive rulebooks like ISBP and UCP are extensive. Concatenating multiple such documents, plus the trade document itself, rapidly exceeds these limits, leading to "request too large" errors.
2.  **Cost Efficiency:** Every token sent to an LLM incurs a cost. Sending entire rulebooks repeatedly for each analysis is extremely expensive and wasteful, as only a small fraction of the rules might be relevant to a specific document or discrepancy.
3.  **Performance and Latency:** Processing a massive input context takes significantly longer for the LLM. This increases latency, making the application slow and unresponsive for the user.
4.  **Accuracy and Hallucination:** When an LLM is given an overly broad context, it can struggle to focus on the most pertinent information. This can lead to less accurate compliance checks, increased "hallucinations" (generating plausible but incorrect findings), and a diluted understanding of the specific rules relevant to the document.
5.  **Scalability:** As the number of rulebooks grows (e.g., adding URC, URDG, etc.), the simple prompt engineering approach would completely break down. A RAG approach, by retrieving only relevant snippets, scales much more effectively.

**The RAG Approach addresses these challenges by:**
- **Intelligent Retrieval:** Only the most semantically similar and relevant sections of the rulebooks are retrieved based on the content of the trade document.
- **Reduced Context Size:** The LLM receives a much smaller, highly focused context, staying within token limits, reducing costs, and improving processing speed.
- **Enhanced Accuracy:** By providing a precise context, the LLM can perform more accurate and targeted compliance checks, minimizing irrelevant information and improving the quality of the generated report.
- **Scalability:** The system can easily incorporate new rulebooks without significantly impacting performance, as the retrieval mechanism ensures only necessary information is passed to the LLM.

## System Architecture

The system follows a modular architecture, primarily consisting of:

1.  **Document Ingestion:** PDF rulebooks (ISBP, UCP) are read and their text content is extracted.
2.  **Rule Chunking and Vectorization:** The extracted rule texts are split into smaller, manageable chunks. Each chunk is then vectorized using a custom TF-IDF implementation. These vectors form an in-memory knowledge base.
3.  **User Document Upload:** Users upload their trade documents (as `.txt` files) via the Streamlit interface.
4.  **Relevant Rule Retrieval (RAG):** When a trade document is uploaded, its content is vectorized. This vector is then used to calculate cosine similarity against all rule chunks in the knowledge base. The top-K most relevant chunks are retrieved.
5.  **LLM Invocation:** The retrieved relevant rule chunks, along with the user's trade document and a specific system prompt, are sent to the Large Language Model (LLM).
6.  **Compliance Analysis:** The LLM analyzes the trade document against the provided relevant rules and generates a structured JSON compliance report.
7.  **Report Display and Storage:** The generated report is displayed in the Streamlit interface and saved as a JSON file in the `output/` directory.

## Technologies and Libraries Used

-   **Python 3.x:** The core programming language for the entire application.
-   **Streamlit:** Used for building the interactive web-based user interface.
    -   *Why:* Provides rapid development of data apps and interactive dashboards with minimal code, making it ideal for a quick prototype and user interaction.
-   **Groq API (llama3-70b-8192):** The Large Language Model (LLM) used for the core compliance analysis and report generation.
    -   *Why:* Groq offers high-performance, low-latency inference for LLMs, which is crucial for a responsive compliance checker. The `llama3-70b-8192` model is chosen for its strong reasoning capabilities and large context window.
-   **PyPDF2:** A Python library used for extracting text content from PDF documents.
    -   *Why:* Essential for ingesting the ISBP and UCP rulebooks, which are typically distributed in PDF format.
-   **`re` (Python's built-in regex module):** Used for text preprocessing (e.g., lowercasing, removing punctuation) in the vectorizer.
    -   *Why:* Standard library, efficient for basic text manipulation.
-   **`collections.Counter`:** Used in the custom TF-IDF implementation for counting word frequencies.
    -   *Why:* Part of Python's standard library, efficient for frequency counting.
-   **`math` module:** Used for mathematical operations (e.g., `log`, `sqrt`) in the custom TF-IDF and cosine similarity calculations.
    -   *Why:* Standard library, provides necessary mathematical functions.
-   **`os` and `json` (Python's built-in modules):** Used for file system operations (reading/writing files, creating directories) and JSON parsing/serialization.
    -   *Why:* Standard library, fundamental for file handling and data interchange.
-   **`dotenv`:** For loading environment variables (like API keys) from a `.env` file.
    -   *Why:* Securely manages sensitive information without hardcoding it into the source code.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Hassan-asim/Trade-Document-Compliance-Checker.git
    cd Trade-Document-Compliance-Checker
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file will be generated or provided containing `streamlit`, `groq`, `PyPDF2`, `python-dotenv`)*

4.  **Set up Groq API Key:**
    -   Obtain an API key from [Groq Console](https://console.groq.com/).
    -   Create a `.env` file in the root directory of the project and add your API key:
        ```
        GROQ_API_KEY=your_groq_api_key_here
        ```

5.  **Place Rule Documents:**
    -   Create a folder named `ISBP rules` in the root directory.
    -   Place your PDF rule documents (e.g., `isbp-745.pdf`, `ISBP-821.pdf`, `UCP 600.pdf`) inside this `ISBP rules` folder. The application will automatically detect and load all PDF files from this directory.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    -   Open your web browser and navigate to the URL displayed in your terminal (usually `http://localhost:8501`).

3.  **Upload Document:**
    -   Use the file uploader in the Streamlit interface to select a `.txt` file of the trade document you wish to analyze.

4.  **View Report:**
    -   The application will process the document against each loaded rulebook independently.
    -   Compliance reports for each rulebook will be displayed separately on the page, detailing discrepancies and compliances.
    -   JSON reports will also be saved in the `output/` directory and can be downloaded directly from the UI.

## Future Enhancements
-   **Advanced Vectorization:** Integrate more sophisticated embedding models (e.g., Sentence Transformers) for better semantic understanding and retrieval.
-   **Persistent Vector Store:** Implement a dedicated vector database (e.g., ChromaDB, FAISS) for efficient storage and retrieval of rule chunks, especially for a larger number of rulebooks.
-   **Dynamic Rule Management:** Allow users to upload and manage rule documents directly through the UI.
-   **Multi-document Analysis:** Enable analysis of multiple trade documents simultaneously.
-   **UI/UX Improvements:** Enhance the user interface for better readability and interaction.
-   **Error Handling and Logging:** More robust error handling and detailed logging for debugging and monitoring.
-   **Test Suite:** Develop a comprehensive test suite to ensure the accuracy and reliability of compliance checks.# Advanced-RAG-Based-Trade-Document-Compliance-Analysis 
