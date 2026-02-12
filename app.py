import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd

# CONFIGURATION
st.set_page_config(page_title="Intelligent Research Hub", layout="wide")

# Securely set up your API Key (In production, use st.secrets)
API_KEY = "AIzaSyCl9M4aMlB8YC9TNoJQgExQ0_ewR11btik" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# HELPER FUNCTIONS
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# SIDEBAR NAVIGATION
st.sidebar.title("🌟 Platform Menu")
app_mode = st.sidebar.selectbox("Choose a Module", ["Home", "AI Research Assistant", "Consulting Chatbot", "Data Analytics"])

# MODULE 1: HOME
if app_mode == "Home":
    st.title("🎓 Intelligent Research & Consulting Platform")
    st.write("Welcome! This platform uses AI to assist with academic research and professional consulting.")
    st.image("https://via.placeholder.com/800x400.png?text=FUNAAB+Research+Hub")
    
# MODULE 2: AI RESEARCH ASSISTANT (PDF Analysis)
elif app_mode == "AI Research Assistant":
    st.header("📄 Research Paper Analyzer")
    uploaded_file = st.file_uploader("Upload a Research PDF (e.g., a past project)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Reading document..."):
            context = extract_pdf_text(uploaded_file)
            st.success("Document Loaded!")
        
        user_question = st.text_input("What would you like to know about this research?")
        
        if user_question:
            prompt = f"Based on this research text: {context[:5000]}, answer this: {user_question}"
            response = model.generate_content(prompt)
            st.markdown("### 🤖 AI Insight:")
            st.write(response.text)

# MODULE 3: CONSULTING CHATBOT
elif app_mode == "Consulting Chatbot":
    st.header("💬 Virtual Academic & Technical Consultant")
    st.info("Ask me for advice on project methodologies, agricultural tips, or career guidance.")
    
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
            # Consulting System Prompt
            full_prompt = f"You are a professional research consultant. Provide expert advice on: {prompt}"
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

# MODULE 4: DATA ANALYTICS
elif app_mode == "Data Analytics":
    st.header("📊 Automated Research Data Analysis")
    data_file = st.file_uploader("Upload your experimental data (CSV)", type="csv")
    
    if data_file:
        df = pd.read_csv(data_file)
        st.write("### Data Preview", df.head())
        
        column_to_plot = st.selectbox("Select a column to visualize", df.columns)
        st.line_chart(df[column_to_plot])
        
        if st.button("Generate Statistical Summary"):
            st.write(df.describe())
