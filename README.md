# Trade Document Compliance Checker

A sophisticated AI-powered Trade Document Compliance Checker that validates trade documents against international banking rules (ISBP, UCP) using a Retrieval-Augmented Generation (RAG) pipeline. The system provides detailed compliance reports with specific rule citations and supports multiple LLM providers for reliability.

## 🚀 Features

- **Multi-LLM Support**: Automatic fallback between GLM-4.5-Flash and Groq (Llama3-70B-8192) for reliable processing
- **Smart Rule Management**: Configurable rule sets with document-specific and general rule categories
- **RAG-Powered Analysis**: Intelligent retrieval of relevant rule sections for accurate compliance checking
- **Multiple Document Types**: Supports Bill of Lading, Commercial Invoice, Packing List, Covering Schedule, DHL Receipt, and Shipment Advice
- **Detailed Compliance Reports**: JSON-structured reports with specific rule citations and compliance/discrepancy classifications
- **Modern Web Interface**: Streamlit-based UI with real-time processing and visual feedback
- **Efficient Processing**: 93% reduction in token usage through intelligent RAG implementation

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Rule Loading   │    │   RAG Pipeline  │
│   Upload        │───▶│   & Management   │───▶│   (Vectorizer)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   LLM Service    │    │   Compliance    │
                       │  (GLM + Groq)   │◀───│   Analysis      │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Fallback       │    │   Report        │
                       │   Mechanism      │    │   Generation    │
                       └──────────────────┘    └─────────────────┘
```

## 🛠️ Technologies & Dependencies

### Core Dependencies
- **Python 3.x**: Core programming language
- **Streamlit**: Web interface framework
- **PyPDF2**: PDF text extraction
- **python-dotenv**: Environment variable management
- **requests**: HTTP client for API calls

### LLM Providers
- **GLM-4.5-Flash** (Primary): High-performance Chinese LLM via Zhipu AI
- **Groq Llama3-70B-8192** (Fallback): Fast inference via Groq API

### AI/ML Components
- **Custom TF-IDF Vectorizer**: Efficient text similarity calculation
- **Cosine Similarity**: Rule relevance scoring
- **Chunking System**: Optimal text processing for large documents

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── rag_llm_pipeline.py            # Core RAG and LLM pipeline
├── vectorizer.py                  # Custom TF-IDF vectorization
├── llm_service.py                 # LLM service with fallback
├── glm_llm.py                     # GLM LLM client implementation
├── rules_config.json              # Rule configuration and mapping
├── system_prompt.md               # Main compliance analysis prompt
├── system_prompt_doc_type.md      # Document type detection prompt
├── system_prompt_rule_categorizer.md # Rule categorization prompt
├── requirements.txt               # Python dependencies
├── ISBP rules/                    # Rule documents directory
│   ├── General_Rules_Common.pdf
│   ├── Bill_of_Lading_Rules.pdf
│   ├── Commercial_Invoice_Validation_Rules.pdf
│   ├── Packing_List_Validation_Rules.pdf
│   ├── Covering_Schedule_Validation_Rules.pdf
│   ├── DHL_Receipt_Validation_Rules.pdf
│   ├── Shipment_Advice_Validation_Rules.pdf
│   ├── isbp-745.pdf
│   ├── isbp-821.pdf
│   └── UCP-600.pdf
├── output/                        # Generated compliance reports
├── transcribe_docs/               # Document transcriptions
└── OCR/                          # OCR processing results
```

## ⚙️ Configuration

### Rule Configuration (`rules_config.json`)
The system uses a smart rule loading mechanism:

```json
{
  "general_rules": ["General_Rules_Common.pdf"],
  "document_specific_rules": {
    "BILL OF LADING": ["Bill_of_Lading_Rules.pdf"],
    "COMMERCIAL INVOICE": ["Commercial_Invoice_Validation_Rules.pdf"],
    "PACKING LIST": ["Packing_List_Validation_Rules.pdf"],
    "COVERING SCHEDULE": ["Covering_Schedule_Validation_Rules.pdf"],
    "DHL RECEIPT": ["DHL_Receipt_Validation_Rules.pdf"],
    "SHIPMENT ADVICE": ["Shipment_Advice_Validation_Rules.pdf"]
  }
}
```

### Environment Variables
Create a `.env` file with your API keys:

```bash
# Primary LLM (GLM-4.5-Flash)
GLM_API_KEY=your_glm_api_key_here

# Fallback LLM (Groq)
GROQ_API_KEY=your_groq_api_key_here
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "1st task"
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
- Obtain API keys from [Zhipu AI Console](https://console.zhipuai.cn/) and [Groq Console](https://console.groq.com/)
- Create `.env` file with your keys

### 5. Prepare Rule Documents
- Place your PDF rule documents in the `ISBP rules/` directory
- Ensure filenames match those referenced in `rules_config.json`

## 💻 Usage

### 1. Start the Application
```bash
streamlit run app.py
```

### 2. Access the Web Interface
- Open your browser to `http://localhost:8501`
- The interface will show loaded rules and configuration

### 3. Upload Documents
- Use the file uploader to select trade documents
- Supported formats: `.txt`, `.pdf` (with OCR processing)
- The system automatically detects document type

### 4. View Results
- Real-time processing with progress indicators
- Separate compliance reports for each rule set
- Download JSON reports for further analysis

## 🔍 How It Works

### 1. Document Processing
- Document type detection using AI
- Text extraction and preprocessing
- Chunking for optimal processing

### 2. Rule Retrieval (RAG)
- TF-IDF vectorization of document and rules
- Cosine similarity calculation
- Top-K most relevant rule sections retrieval

### 3. Compliance Analysis
- LLM processes document against retrieved rules
- Structured JSON output with specific citations
- Classification into compliances and discrepancies

### 4. Report Generation
- Comprehensive compliance summary
- Rule-specific analysis results
- Exportable JSON format

## 📊 Performance Features

- **Token Efficiency**: 93% reduction in token usage (225K → 11K chars)
- **Fast Processing**: Intelligent chunking and vectorization
- **Reliable Fallback**: Automatic LLM provider switching
- **Memory Optimization**: Selective rule loading based on document type

## 🔧 Advanced Features

### Multi-LLM Fallback System
- Primary: GLM-4.5-Flash for high-quality analysis
- Fallback: Groq Llama3-70B-8192 for reliability
- Automatic switching on API failures

### Smart Rule Management
- Document-specific rule sets
- General rules for universal compliance
- Configurable rule loading

### Enhanced User Experience
- Real-time processing feedback
- Visual progress indicators
- Configuration transparency
- Error handling and debugging

## 🚧 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure `.env` file contains valid API keys
2. **PDF Reading Issues**: Check PDF file integrity and permissions
3. **Memory Issues**: Large rule documents may require more RAM

### Debug Mode
- Check console output for detailed processing information
- Review `output/` directory for generated reports
- Verify rule configuration in `rules_config.json`

## 🔮 Future Enhancements

- [ ] Support for additional document types
- [ ] Integration with more LLM providers
- [ ] Advanced rule customization interface
- [ ] Batch processing capabilities
- [ ] API endpoint for external integrations
- [ ] Enhanced visualization and reporting

## 📝 License

This project is designed for educational and commercial use in trade document compliance checking.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests are added for new features
- Documentation is updated
- API keys are never committed

---

**Built with ❤️ for the international trade finance community**

