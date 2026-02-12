import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
import time  # New: for waiting between retries

# --- CONFIGURATION ---
st.set_page_config(page_title="Intelligent Research Hub", layout="wide")

# Securely set up your API Key
API_KEY = "AIzaSyCl9M4aMlB8YC9TNoJQgExQ0_ewR11btik" 
genai.configure(api_key=API_KEY)

# Use 'gemini-1.5-flash' for higher free-tier limits
model = genai.GenerativeModel('gemini-1.5-flash')

# --- HELPER FUNCTIONS ---
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def safe_generate_content(prompt):
    """Function to handle 'Resource Exhausted' error with retries"""
    max_retries = 3
    for i in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e): # This is the Resource Exhausted error
                if i < max_retries - 1:
                    wait_time = (i + 1) * 5
                    st.warning(f"Server busy. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return "❌ Error: API Quota exhausted. Please wait 1 minute and try again."
            else:
                return f"❌ An unexpected error occurred: {str(e)}"

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌟 Platform Menu")
app_mode = st.sidebar.selectbox("Choose a Module", ["Home", "AI Research Assistant", "Consulting Chatbot", "Data Analytics"])

# MODULE 1: HOME
if app_mode == "Home":
    st.title("🎓 Intelligent Research & Consulting Platform")
    st.write("Welcome! This platform uses AI to assist with academic research and professional consulting.")
    
# MODULE 2: AI RESEARCH ASSISTANT (PDF Analysis)
elif app_mode == "AI Research Assistant":
    st.header("📄 Research Paper Analyzer")
    uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Reading document..."):
            context = extract_pdf_text(uploaded_file)
            st.success("Document Loaded!")
        
        user_question = st.text_input("What would you like to know about this research?")
        
        if user_question:
            with st.spinner("Analyzing..."):
                prompt = f"Based on this research: {context[:5000]}, answer this: {user_question}"
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
                full_prompt = f"You are a professional research consultant. Answer: {prompt}"
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
