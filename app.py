import streamlit as st
import os
import json
import PyPDF2
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_llm_pipeline import RAGLLMPipeline

@st.cache_data
def read_pdf_text(pdf_path: str) -> str:
    """Reads text content from a PDF file, cleans it, and caches the result."""
    if not os.path.exists(pdf_path):
        st.error(f"Fatal Error: PDF file not found at {pdf_path}")
        return None
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_content = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                # Basic cleaning: remove multiple newlines and excessive whitespace
                page_text = re.sub(r'\n+', '\n', page_text).strip()
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                # Add page separator for better readability
                text_content.append(f"==Start of OCR for page {i+1}==\n{page_text}\n==End of OCR for page {i+1}==")
        return "\n\n".join(text_content)
    except Exception as e:
        st.error(f"Fatal Error: Could not read or parse the PDF file at {pdf_path}. Error: {e}")
        return None

# Page config
st.set_page_config(
    page_title="Validation Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = "Light Theme"

# Get the base directory path
base_path = os.path.dirname(os.path.abspath(__file__))

# Load rules configuration
rules_config_path = os.path.join(base_path, 'rules_config.json')
try:
    with open(rules_config_path, 'r') as f:
        rules_config = json.load(f)
    general_rules_filenames = rules_config.get('general_rules', [])
    document_specific_rules_map = rules_config.get('document_specific_rules', {})
except FileNotFoundError:
    st.error("❌ rules_config.json not found. Please ensure the configuration file exists.")
    st.stop()
except json.JSONDecodeError:
    st.error("❌ Invalid JSON in rules_config.json. Please check the file format.")
    st.stop()

# CSS Styles - Always applied
st.markdown("""
<style>
    /* Base styles that always apply */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        color: white !important;
    }
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    .main-description {
        font-size: 1rem;
        opacity: 0.8;
        color: white !important;
    }
    
    /* Sidebar always keeps its styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .stSidebar h3,
    .stSidebar h4,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4 {
        color: white !important;
    }
    
    /* Neutral metric cards */
    .metric-card {
        background: #e9ecef !important;
        color: #212529 !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Success message fade out */
    .success-msg {
        opacity: 1;
        transition: opacity 0.5s ease-out;
    }
    .success-msg.fade-out {
        opacity: 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header with gradient background
st.markdown("""
<div class="main-header">
    <div class="main-title">🔍 Validation Checker</div>
    <div class="main-subtitle">for compliance analysis</div>
    <div class="main-description">Advanced RAG-Based Trade Document Compliance Analysis</div>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>🎨 Choose Theme</h2></div>', unsafe_allow_html=True)
    
    # Theme selector
    theme_option = st.selectbox(
        "Choose Theme",
        ["Light Theme", "Dark Theme", "Auto"],
        index=["Light Theme", "Dark Theme", "Auto"].index(st.session_state.theme),
        key="theme_selector",
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state.theme = theme_option
    
    st.markdown("---")
    
    st.markdown('<div class="sidebar-header"><h3>⚙️ Settings & Configuration</h3></div>', unsafe_allow_html=True)
    
    st.markdown("#### 📋 Rules Configuration")
    st.markdown("#### 📄 General Rules")
    for rule in general_rules_filenames:
        st.markdown(f"• `{rule}`")

    st.markdown("#### 🎯 Document-Specific Rules")
    for doc_type, rules in document_specific_rules_map.items():
        with st.expander(f"📑 {doc_type}", expanded=False):
            for rule in rules:
                st.markdown(f"• `{rule}`")

    st.markdown("---")
    st.info("💡 **How it works**: Upload a document, and the system will automatically detect its type and apply the relevant rules for validation.")
    
    st.markdown("#### 🔧 System Status")
    st.success("✅ RAG Pipeline: Active")
    st.success("✅ API Services: Ready") 
    st.info(f"📊 Document Types: {len(document_specific_rules_map)}")

# Apply theme-specific styles
if theme_option == "Dark Theme":
    st.markdown("""
    <style>
    /* Dark theme - Main page only */
    .stApp > .main {
        background-color: #1e1e1e !important;
    }
    .stApp .main .block-container {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    .stApp .main .stMarkdown,
    .stApp .main .stText,
    .stApp .main p,
    .stApp .main span,
    .stApp .main div[data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }
    .stApp .main .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d !important;
    }
    .stApp .main .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    .stApp .main .stExpander {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    .stApp .main .stExpander .streamlit-expanderHeader,
    .stApp .main .stExpander .streamlit-expanderContent {
        color: #ffffff !important;
    }
    /* Keep gradient headers as they are */
    .main-header *, .sidebar-header * {
        color: white !important;
    }
    /* Keep metric cards neutral */
    .metric-card h3, .metric-card h4 {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
elif theme_option == "Light Theme":
    st.markdown("""
    <style>
    /* Light theme - Main page only */
    .stApp > .main {
        background-color: #ffffff !important;
    }
    .stApp .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stApp .main .stMarkdown,
    .stApp .main .stText,
    .stApp .main p,
    .stApp .main span,
    .stApp .main div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    .stApp .main .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
    }
    .stApp .main .stTabs [data-baseweb="tab"] {
        color: #000000 !important;
    }
    .stApp .main .stExpander {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    .stApp .main .stExpander .streamlit-expanderHeader,
    .stApp .main .stExpander .streamlit-expanderContent {
        color: #000000 !important;
    }
    /* Keep gradient headers as they are */
    .main-header *, .sidebar-header * {
        color: white !important;
    }
    /* Keep metric cards neutral */
    .metric-card h3, .metric-card h4 {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)

else:  # Auto theme
    st.markdown("""
    <style>
    /* Auto theme - follows system preference */
    @media (prefers-color-scheme: dark) {
        .stApp > .main {
            background-color: #1e1e1e !important;
        }
        .stApp .main .block-container {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        .stApp .main .stMarkdown,
        .stApp .main .stText,
        .stApp .main p,
        .stApp .main span,
        .stApp .main div[data-testid="stMarkdownContainer"] {
            color: #ffffff !important;
        }
    }
    @media (prefers-color-scheme: light) {
        .stApp > .main {
            background-color: #ffffff !important;
        }
        .stApp .main .block-container {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stApp .main .stMarkdown,
        .stApp .main .stText,
        .stApp .main p,
        .stApp .main span,
        .stApp .main div[data-testid="stMarkdownContainer"] {
            color: #000000 !important;
        }
    }
    /* Keep gradient headers as they are */
    .main-header *, .sidebar-header * {
        color: white !important;
    }
    /* Keep metric cards neutral */
    .metric-card h3, .metric-card h4 {
        color: #212529 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load all rule texts
all_rule_texts = {}
rules_dir = os.path.join(base_path, 'ISBP rules')
success_container = st.container()

if os.path.isdir(rules_dir):
    referenced_files = set(general_rules_filenames)
    for doc_type, rule_files in document_specific_rules_map.items():
        referenced_files.update(rule_files)
    
    success_messages = []
    
    for rule_file in os.listdir(rules_dir):
        if rule_file.endswith('.pdf') and rule_file in referenced_files:
            rule_path = os.path.join(rules_dir, rule_file)
            rule_text = read_pdf_text(rule_path)
            if rule_text:
                all_rule_texts[rule_file] = rule_text
                success_messages.append(f"✅ {rule_file} loaded successfully")
    
    # Display success messages with auto-disappear
    if success_messages:
        with success_container:
            success_placeholder = st.empty()
            with success_placeholder.container():
                st.markdown('<div class="success-msg" id="success-messages">', unsafe_allow_html=True)
                for msg in success_messages:
                    st.success(msg)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <script>
            setTimeout(function() {
                const successContainer = document.getElementById('success-messages');
                if (successContainer) {
                    successContainer.classList.add('fade-out');
                    setTimeout(function() {
                        successContainer.style.display = 'none';
                    }, 500);
                }
            }, 5000);
            </script>
            """, unsafe_allow_html=True)
    
    missing_files = referenced_files - set(all_rule_texts.keys())
    if missing_files:
        st.error(f"❌ Missing rule files: {', '.join(missing_files)}")
        st.info("Please ensure all files referenced in rules_config.json are present in the ISBP rules folder.")
else:
    st.error(f"Error: ISBP rules directory not found at {rules_dir}.")
    st.stop()

# Main App Logic
if all_rule_texts:
    st.markdown("### 📤 Upload Document for Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a .txt file for compliance analysis", 
            type="txt",
            help="Upload your trade document in .txt format for automated compliance checking"
        )
    
    with col2:
        if uploaded_file:
            st.markdown("""
            <div class="metric-card">
                <strong>📄 File Info</strong><br>
                <small>Ready for analysis</small>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode("utf-8")
            file_name = uploaded_file.name
            
            st.markdown("---")
            
            # Document type detection
            with st.spinner("🔍 Detecting document type..."):
                # Cache the pipeline instance across reruns
                @st.cache_resource(show_spinner=False)
                def get_pipeline():
                    return RAGLLMPipeline()

                pipeline = get_pipeline()

                @st.cache_data(show_spinner=False)
                def cached_detect_document_type(content: str) -> str:
                    return pipeline.detect_document_type(content)

                doc_type_result = cached_detect_document_type(file_content)
            
            if doc_type_result and doc_type_result != "UNKNOWN":
                detected_doc_type = doc_type_result
                
                # Display detected document type
                st.markdown(f"""
                <div class="metric-card" style="
                    background: linear-gradient(135deg, #28a745 0%, #20c997 50%, #17a2b8 100%);
                    border: none;
                    color: white;
                    box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
                ">
                    <h4 style="color: white; margin: 0; font-weight: 600;">📋 Detected Document Type</h4>
                    <h3 style="margin: 0.5rem 0 0 0; color: white; font-weight: 700;">{detected_doc_type}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Determine validation sets
                validation_sets = []
                
                # Add general rules
                general_rules_available = [rule for rule in general_rules_filenames if rule in all_rule_texts]
                for rule_file in general_rules_available:
                    validation_sets.append((rule_file, all_rule_texts[rule_file]))
                
                # Add document-specific rules
                specific_rules_filenames = document_specific_rules_map.get(detected_doc_type, [])
                specific_rules_available = [rule for rule in specific_rules_filenames if rule in all_rule_texts]
                for rule_file in specific_rules_available:
                    validation_sets.append((rule_file, all_rule_texts[rule_file]))
                
                if validation_sets:
                    # Analysis Summary
                    st.markdown("### 📊 Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #2a5298;">
                            <h4 style="color: #2a5298; margin: 0;">📋 Document Type</h4>
                            <h3 style="margin: 0.5rem 0 0 0; color: #212529;">{detected_doc_type}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #28a745;">
                            <h4 style="color: #28a745; margin: 0;">📚 Rule Sets</h4>
                            <h3 style="margin: 0.5rem 0 0 0; color: #212529;">{len(validation_sets)}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col3:
                        total_rules = len(general_rules_available) + len(specific_rules_filenames)
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #fd7e14;">
                            <h4 style="color: #fd7e14; margin: 0;">📄 Total Rules</h4>
                            <h3 style="margin: 0.5rem 0 0 0; color: #212529;">{total_rules}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col4:
                        doc_size = f"{len(file_content):,} chars"
                        st.markdown(f"""
                        <div class="metric-card" style="border-left-color: #6f42c1;">
                            <h4 style="color: #6f42c1; margin: 0;">📐 Document Size</h4>
                            <h3 style="margin: 0.5rem 0 0 0; color: #212529;">{doc_size}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Rules Applied Section
                    st.markdown("### 📋 Rules Applied")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🌐 General Rules:**")
                        for rule in general_rules_available:
                            st.markdown(f"• `{rule}`")
                    
                    with col2:
                        st.markdown(f"**🎯 {detected_doc_type} Specific Rules:**")
                        for rule in specific_rules_available:
                            st.markdown(f"• `{rule}`")
                    
                    # Process each validation set
                    st.markdown("### 📝 Compliance Analysis Results")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Run analyses in parallel without caching results
                    total = len(validation_sets)
                    completed = 0
                    progress_bar.progress(0 if total else 1)

                    all_results_map = {}
                    if total:
                        with ThreadPoolExecutor(max_workers=min(4, total)) as executor:
                            future_to_idx = {}
                            for idx, (rfname, rtext) in enumerate(validation_sets):
                                status_text.text(f"🔍 Analyzing against {rfname}...")
                                fut = executor.submit(
                                    pipeline.process_document_for_compliance,
                                    {"content": file_content, "filename": file_name},
                                    rtext,
                                    rfname
                                )
                                future_to_idx[fut] = (idx, rfname)
                            for fut in as_completed(future_to_idx):
                                idx, rfname = future_to_idx[fut]
                                try:
                                    result = fut.result()
                                except Exception:
                                    result = None
                                if result:
                                    all_results_map[idx] = (rfname, result)
                                completed += 1
                                progress_bar.progress(completed / total if total else 1)

                    status_text.empty()
                    progress_bar.empty()

                    # Reassemble results in original order
                    all_results = [all_results_map[i] for i in sorted(all_results_map.keys())]
                    
                    # Display results
                    if all_results:
                        for rules_filename, result in all_results:
                            with st.expander(f"📋 Analysis against {rules_filename}", expanded=True):
                                try:
                                    if isinstance(result, str):
                                        compliance_data = json.loads(result)
                                    else:
                                        compliance_data = result
                                    
                                    compliance_report = compliance_data.get("compliance_report", [])
                                    
                                    if compliance_report and len(compliance_report) > 0:
                                        report = compliance_report[0]
                                        discrepancies = report.get("discrepancies", [])
                                        compliances = report.get("compliances", [])
                                        
                                        # Summary metrics
                                        metric_cols = st.columns(3)
                                        
                                        with metric_cols[0]:
                                            st.markdown(f"""
                                            <div class="metric-card" style="border-left-color: #dc3545; text-align: center;">
                                                <h4 style="color: #dc3545; margin: 0;">❌ Discrepancies</h4>
                                                <h2 style="margin: 0.5rem 0 0 0; color: #212529;">{len(discrepancies)}</h2>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with metric_cols[1]:
                                            st.markdown(f"""
                                            <div class="metric-card" style="border-left-color: #28a745; text-align: center;">
                                                <h4 style="color: #28a745; margin: 0;">✅ Compliances</h4>
                                                <h2 style="margin: 0.5rem 0 0 0; color: #212529;">{len(compliances)}</h2>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        with metric_cols[2]:
                                            total_checks = len(discrepancies) + len(compliances)
                                            compliance_rate = (len(compliances) / total_checks * 100) if total_checks > 0 else 0
                                            st.markdown(f"""
                                            <div class="metric-card" style="border-left-color: #6f42c1; text-align: center;">
                                                <h4 style="color: #6f42c1; margin: 0;">📊 Compliance Rate</h4>
                                                <h2 style="margin: 0.5rem 0 0 0; color: #212529;">{compliance_rate:.1f}%</h2>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        # Detailed results
                                        tab1, tab2 = st.tabs(["❌ Discrepancies", "✅ Compliances"])
                                        
                                        with tab1:
                                            if discrepancies:
                                                for idx, disc in enumerate(discrepancies, 1):
                                                    with st.expander(f"❌ Discrepancy {idx}", expanded=False):
                                                        st.markdown(f"**Finding:** {disc.get('finding', 'No finding specified')}")
                                                        st.markdown(f"**Rule:** `{disc.get('rule', 'No rule specified')}`")
                                            else:
                                                st.success("🎉 No discrepancies found!")
                                        
                                        with tab2:
                                            if compliances:
                                                for idx, comp in enumerate(compliances, 1):
                                                    with st.expander(f"✅ Compliance {idx}", expanded=False):
                                                        st.markdown(f"**Finding:** {comp.get('finding', 'No finding specified')}")
                                                        st.markdown(f"**Rule:** `{comp.get('rule', 'No rule specified')}`")
                                            else:
                                                st.info("ℹ️ No compliance items found.")
                                        
                                        # Download section
                                        st.markdown("#### 💾 Download Results")
                                        json_str = json.dumps(compliance_data, indent=2)
                                        st.download_button(
                                            label="📥 Download JSON Report",
                                            data=json_str,
                                            file_name=f"{file_name}_{rules_filename}_compliance_report.json",
                                            mime="application/json"
                                        )
                                    
                                    else:
                                        st.warning("⚠️ No compliance report generated.")
                                
                                except json.JSONDecodeError as e:
                                    st.error(f"❌ Error parsing compliance results: {e}")
                                    st.text("Raw response:")
                                    st.code(str(result))
                                except Exception as e:
                                    st.error(f"❌ Unexpected error: {e}")
                                    st.text("Raw response:")
                                    st.code(str(result))
                    else:
                        st.error("❌ No analysis results generated.")
                else:
                    st.error("❌ No applicable rules found for this document type.")
            else:
                st.error("❌ Failed to detect document type.")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
else:
    st.error("❌ No rule texts loaded. Please check your configuration and rule files.")