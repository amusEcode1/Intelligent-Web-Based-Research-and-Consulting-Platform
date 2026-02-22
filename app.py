import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
import time
from google.api_core import exceptions

# --- CONFIGURATION ---
st.set_page_config(page_title="Intelligent Research Hub", layout="wide")

# Securely set up your API Key
API_KEY = "AIzaSyACjodqLxCbN0SaZO5VI9eKC2kkb9gbNB4" 
genai.configure(api_key=API_KEY)

# Use 'gemini-1.5-flash' - it has the most generous free limits in 2026
model = genai.GenerativeModel('gemini-2.0-flash')

# --- HELPER FUNCTIONS ---
def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def safe_generate_content(prompt):
    """
    Fixed Retry Logic: Uses exponential backoff to stop '429 Resource Exhausted' errors.
    """
    max_retries = 4
    for i in range(max_retries):
        try:
            # Mandatory 2-second pause to prevent 'Burst' limit errors
            time.sleep(2) 
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            if i < max_retries - 1:
                # Wait progressively longer: 10s, 20s, 40s
                wait_time = (i + 1) * 10 
                st.warning(f"⚠️ Google Servers busy. Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
            else:
                return "❌ API Quota fully exhausted. Please wait 2 minutes and refresh the page."
        except Exception as e:
            return f"❌ AI Error: {str(e)}"

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌟 Platform Menu")
app_mode = st.sidebar.selectbox("Choose a Module", ["Home", "AI Research Assistant", "Consulting Chatbot", "Data Analytics"])

# MODULE 1: HOME
if app_mode == "Home":
    st.title("🎓 Intelligent Research & Consulting Platform")
    st.write("Welcome! This platform uses AI to assist with academic research and professional consulting.")

# MODULE 2: AI RESEARCH ASSISTANT
elif app_mode == "AI Research Assistant":
    st.header("📄 Research Paper Analyzer")
    uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Reading document..."):
            context = extract_pdf_text(uploaded_file)
            st.success("Document Loaded!")
        
        user_question = st.text_input("What would you like to know about this research?")
        
        if user_question:
            with st.spinner("Analyzing (this may take a moment)..."):
                # TRUNCATION: We only send the first 4000 characters to stay under the TPM limit
                prompt = f"Using this research context: {context[:4000]}\n\nQuestion: {user_question}"
                answer = safe_generate_content(prompt)
                st.markdown("### 🤖 AI Insight:")
                st.write(answer)

# MODULE 3: CONSULTING CHATBOT
elif app_mode == "Consulting Chatbot":
    st.header("💬 Virtual Academic & Technical Consultant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting..."):
                full_prompt = f"System: You are an expert academic consultant. User asks: {prompt}"
                answer = safe_generate_content(full_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# MODULE 4: DATA ANALYTICS
elif app_mode == "Data Analytics":
    st.header("📊 Automated Research Data Analysis")
    data_file = st.file_uploader("Upload your experimental data (CSV)", type="csv")
    
    if data_file:
        df = pd.read_csv(data_file)
        st.write("### Data Preview", df.head())
        column_to_plot = st.selectbox("Select a column to visualize", df.columns)
        st.line_chart(df[column_to_plot])
