import streamlit as st
import pandas as pd
import json
from PyPDF2 import PdfReader
from groq import Groq
import io
import logging
from datetime import datetime
import re
import os

# -------------------------
# CONFIG & STYLING
# -------------------------
st.set_page_config(
    page_title="Smart Invoice Extractor Pro",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #2E4053 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        color: #5D6D7E !important;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .success-box {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .file-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# INITIALIZATION
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SAFE API KEY HANDLING - NO KEY EXPOSED!
try:
    # This works when deployed on Streamlit Cloud with secrets
    api_key = st.secrets["GROQ_API_KEY"]
except:
    # This works locally if you set environment variable
    api_key = os.environ.get('GROQ_API_KEY')
    # If no key found, show demo mode
    if not api_key:
        st.sidebar.warning("🔒 Demo Mode - Limited functionality")
        st.sidebar.info("Get free API key from: https://console.groq.com")
        api_key = "demo-key-placeholder"  # This won't work for real processing

# Initialize Groq client safely
try:
    client = Groq(api_key=api_key) if api_key != "demo-key-placeholder" else None
except:
    client = None
    st.sidebar.error("❌ API key not configured properly")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def read_pdf(file):
    """Extract text from a PDF file"""
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def extract_invoice_data(text):
    """Use AI to extract structured invoice data with professional enhancements"""
    # Check if we have a valid client (API key)
    if client is None:
        st.error("❌ API key not configured. Please add your Groq API key in Streamlit secrets.")
        return json.dumps({"error": "API key not configured"})
    
    prompt = f"""
    You are an expert invoice data extraction AI for accounting professionals.
    Extract the following information from the invoice text below:

    REQUIRED FIELDS:
    - vendor_name: The company/person who issued the invoice (required)
    - customer_name: The customer receiving the invoice (required)
    - invoice_date: The invoice date in YYYY-MM-DD format (required)
    - invoice_number: The invoice ID/number (required)
    - subtotal_amount: Subtotal before taxes (if shown)
    - tax_amount: Tax/VAT amount (if shown)
    - discount_amount: Any discount amount (if shown, as positive number)
    - total_amount: The final invoice total (required)
    - amount_due: Amount outstanding (default to total_amount if not specified)
    - currency: ISO currency code (USD, EUR, GBP, etc.) (required)
    - payment_status: One of: ['unpaid', 'paid', 'partial'] (infer from context)
    - due_date: Payment due date if specified (YYYY-MM-DD format)
    - payment_terms: Payment terms description

    BUSINESS LOGIC RULES:
    1. amount_due MUST equal total_amount if no payments are mentioned
    2. Convert all dates to ISO format (YYYY-MM-DD)
    3. For discounts: extract as positive numbers, subtract from subtotal
    4. If tax is not separate, set tax_amount to 0
    5. payment_status: 'unpaid' if amount_due == total_amount, 'paid' if amount_due == 0, 'partial' otherwise

    Return ONLY a valid JSON object with these exact keys.
    Use null for missing optional fields.

    INVOICE TEXT:
    {text}
    """

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        full_response = response.choices[0].message.content

        # Extract JSON safely
        start_index = full_response.find('{')
        end_index = full_response.rfind('}') + 1
        json_string = full_response[start_index:end_index] if start_index != -1 else full_response

        return json_string
    except Exception as e:
        return json.dumps({"error": f"API call failed: {str(e)}"})

def validate_invoice_data(data_dict):
    """Comprehensive validation and cleaning of invoice data"""
    try:
        # Convert amounts to float
        amounts = ['subtotal_amount', 'tax_amount', 'discount_amount', 'total_amount', 'amount_due']
        for key in amounts:
            if data_dict.get(key):
                data_dict[key] = float(data_dict[key])
        
        subtotal = data_dict.get("subtotal_amount") or 0
        tax = data_dict.get("tax_amount") or 0
        discount = data_dict.get("discount_amount") or 0
        total = data_dict.get("total_amount") or 0
        due = data_dict.get("amount_due") or total

        # Set payment status
        if due == 0:
            data_dict["payment_status"] = "paid"
        elif due < total:
            data_dict["payment_status"] = "partial"
        else:
            data_dict["payment_status"] = "unpaid"
            
    except Exception as e:
        logger.warning(f"Data validation note: {e}")
    
    return data_dict

def process_invoices(uploaded_files):
    """Process multiple PDFs with professional error handling"""
    data = []
    processed_files = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        try:
            status_text.text(f"🔍 Processing {i+1}/{len(uploaded_files)}: {file.name}")
            pdf_text = read_pdf(file)
            extracted_json = extract_invoice_data(pdf_text)
            data_dict = json.loads(extracted_json)
            
            # Skip if API error
            if "error" in data_dict:
                st.error(f"API Error: {data_dict['error']}")
                continue
                
            data_dict = validate_invoice_data(data_dict)
            data_dict['source_file'] = file.name
            data.append(data_dict)
            processed_files += 1
            logger.info(f"Successfully processed {file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            st.error(f"⚠️ Failed to process {file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text(f"✅ Processed {processed_files}/{len(uploaded_files)} files successfully")
    return pd.DataFrame(data)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("⚙️ Invoice Extractor Pro")
    st.markdown("---")
    
    st.subheader("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.7, 1.0, 0.9)
    auto_validate = st.checkbox("Auto-validate Data", value=True)
    show_analytics = st.checkbox("Show Analytics Dashboard", value=True)
    
    st.markdown("---")
    st.subheader("API Configuration")
    if client is None:
        st.error("❌ Groq API key not configured")
        st.info("Add your key in Streamlit Secrets: GROQ_API_KEY = your_key_here")
    else:
        st.success("✅ Groq API configured successfully")
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **Smart Invoice Extractor Pro** uses AI to automatically extract and validate invoice data from PDF files.
    
    Features:
    - Multi-file processing
    - Data validation
    - Analytics dashboard
    - Professional reporting
    """)
    
    st.markdown("---")
    st.caption("v2.0 • Powered by Groq AI")

# -------------------------
# MAIN UI
# -------------------------
st.markdown("<h1 class='main-header'>📑 Smart Invoice Extractor Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>AI-powered invoice data extraction with professional analytics</p>", unsafe_allow_html=True)

# File upload section
uploaded_files = st.file_uploader(
    "**Upload Invoice PDFs**", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Select one or multiple invoice PDF files for processing"
)

if uploaded_files:
    # Check if API is configured
    if client is None:
        st.error("""
        ❌ Groq API key not configured. 
        
        To use this app:
        1. Get a free API key from https://console.groq.com
        2. Add it to Streamlit Secrets as GROQ_API_KEY
        3. Redeploy your app
        """)
        st.stop()
    
    # Display uploaded files
    st.subheader("📁 Uploaded Files")
    for file in uploaded_files:
        st.markdown(f"""
        <div class="file-card">
            <strong>{file.name}</strong> • {file.size // 1024} KB
        </div>
        """, unsafe_allow_html=True)
    
    # Process invoices
    with st.spinner("🚀 Processing invoices with AI-powered extraction..."):
        df = process_invoices(uploaded_files)
        if auto_validate and len(df) > 0:
            df = df.apply(validate_invoice_data, axis=1)

    if len(df) > 0:
        # Success message
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Extraction Completed Successfully!</h3>
            <p>Processed {len(df)} invoices with {len(df.columns)} data fields each</p>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Detailed Data", "📈 Analytics", "💾 Export"])

        with tab1:
            st.subheader("Invoice Summary Dashboard")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Total Invoices</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_amount = df['total_amount'].sum()
                currency = df['currency'].iloc[0] if len(df) > 0 else ''
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_amount:,.2f}</div>
                    <div class="metric-label">Total Amount ({currency})</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                outstanding = df['amount_due'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{outstanding:,.2f}</div>
                    <div class="metric-label">Outstanding ({currency})</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                paid_count = len(df[df['payment_status'] == 'paid'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{paid_count}</div>
                    <div class="metric-label">Paid Invoices</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick overview
            st.subheader("Quick Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Vendor Distribution**")
                vendor_counts = df['vendor_name'].value_counts()
                st.dataframe(vendor_counts, use_container_width=True)
            
            with col2:
                st.write("**Payment Status**")
                status_df = df['payment_status'].value_counts().reset_index()
                status_df.columns = ['Status', 'Count']
                st.dataframe(status_df, use_container_width=True)

        with tab2:
            st.subheader("Detailed Invoice Data")
            
            # Enhanced dataframe styling
            styled_df = df.style.format({
                'subtotal_amount': '€{:.2f}' if len(df) > 0 and df['currency'].iloc[0] == 'EUR' else '${:.2f}',
                'tax_amount': '€{:.2f}' if len(df) > 0 and df['currency'].iloc[0] == 'EUR' else '${:.2f}',
                'total_amount': '€{:.2f}' if len(df) > 0 and df['currency'].iloc[0] == 'EUR' else '${:.2f}',
                'amount_due': '€{:.2f}' if len(df) > 0 and df['currency'].iloc[0] == 'EUR' else '${:.2f}'
            }).apply(lambda x: ['background: #e8f5e8' if v == 'paid' else 
                               'background: #ffebee' if v == 'unpaid' else 
                               'background: #fff3e0' for v in x], 
                    subset=['payment_status'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Data statistics
            st.subheader("Data Statistics")
            st.json({
                "total_records": len(df),
                "columns": list(df.columns),
                "data_types": {col: str(df[col].dtype) for col in df.columns},
                "missing_values": df.isnull().sum().to_dict()
            })

        with tab3:
            if show_analytics:
                st.subheader("📈 Analytics Dashboard")
                
                if len(df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Payment status - using Streamlit native
                        st.write("**Payment Status Distribution**")
                        status_counts = df['payment_status'].value_counts()
                        st.dataframe(status_counts)
                        
                    with col2:
                        # Vendor distribution - using Streamlit native
                        st.write("**Top Vendors**")
                        vendor_counts = df['vendor_name'].value_counts().head(5)
                        st.dataframe(vendor_counts)
                    
                    # Amounts by vendor - using Streamlit native bar chart
                    st.subheader("Amounts by Vendor")
                    vendor_totals = df.groupby('vendor_name')['total_amount'].sum().sort_values(ascending=False)
                    st.bar_chart(vendor_totals)
                else:
                    st.info("No data available for analytics")

        with tab4:
            st.subheader("Export Data")
            
            export_option = st.radio("Select Export Format:", 
                                    ["All Data", "Summary Report", "Outstanding Invoices Only"])
            
            if export_option == "All Data":
                export_df = df
            elif export_option == "Summary Report":
                export_df = df[['vendor_name', 'invoice_number', 'invoice_date', 
                               'total_amount', 'currency', 'payment_status', 'amount_due']]
            else:
                export_df = df[df['amount_due'] > 0]
            
            st.write(f"**Exporting:** {len(export_df)} records")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Excel export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Invoices')
                st.download_button(
                    "📊 Download Excel", 
                    data=output.getvalue(),
                    file_name="professional_invoice_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download as Excel spreadsheet"
                )
            
            with col2:
                # CSV export
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "📝 Download CSV", 
                    data=csv,
                    file_name="invoice_data.csv",
                    mime="text/csv",
                    help="Download as CSV file"
                )
            
            with col3:
                # JSON export
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    "🔖 Download JSON", 
                    data=json_data,
                    file_name="invoice_data.json",
                    mime="application/json",
                    help="Download as JSON file"
                )
            
            # Preview of export data
            st.subheader("Export Preview")
            st.dataframe(export_df.head(), use_container_width=True)
    else:
        st.warning("No valid invoice data was extracted. Please check your API key configuration.")

else:
    # Welcome screen when no files are uploaded
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        ## 🚀 Getting Started
        
        1. **Upload PDF invoices** using the file uploader
        2. **AI processing** will automatically extract all relevant data
        3. **Review and validate** the extracted information
        4. **Export** in multiple formats (Excel, CSV, JSON)
        
        Supported features:
        - Multi-file batch processing
        - Automatic data validation
        - Professional analytics dashboard
        - Currency detection and conversion
        - Payment status classification
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3379/3379027.png", width=150)
        st.write("")
        st.write("**Supported Formats:**")
        st.write("✅ PDF invoices")
        st.write("✅ Multi-language support")
        st.write("✅ Various invoice layouts")
        
    st.markdown("---")
    st.caption("💡 Tip: Upload multiple invoices at once for batch processing")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("© 2024 Smart Invoice Extractor Pro • Built with Streamlit & Groq AI • v2.0")