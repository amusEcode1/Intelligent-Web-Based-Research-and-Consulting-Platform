!pip install sentence-transformers -q

!pip install pdfplumber python-docx -q

!pip install streamlit -q

import streamlit as st
import pandas as pd
import numpy as np
import docx
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from google.colab import drive
drive.mount('/content/drive')

# Load Model and Data
@st.cache_resource
def load_model():
    return SentenceTransformer("/content/drive/MyDrive/Models/Resume_Screening/sentence_model")

@st.cache_data
def load_data():
    resume_df = pd.read_csv("/content/drive/MyDrive/Models/Resume_Screening/resume_data.csv")
    job_df = pd.read_csv("/content/drive/MyDrive/Models/Resume_Screening/job_data.csv")
    resume_embeddings = np.load("/content/drive/MyDrive/Models/Resume_Screening/resume_embeddings.npy")
    job_embeddings = np.load("/content/drive/MyDrive/Models/Resume_Screening/job_embeddings.npy")
    return resume_df, job_df, resume_embeddings, job_embeddings

model = load_model()
resume_df, job_df, resume_embeddings, job_embeddings = load_data()

nlp = spacy.load("en_core_web_sm")

SKILLS = ["python", "java", "sql", "excel", "tableau", "power bi",
          "machine learning", "nlp", "deep learning", "data analysis",
          "tensorflow", "pytorch", "communication", "leadership", "statistics"]

def extract_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = str(uploaded_file.read(), 'utf-8')
    return text

def extract_skills(text):
    text_lower = text.lower()
    return [skill for skill in SKILLS if skill in text_lower]

def extract_experience(text):
    doc = nlp(text)
    exp_sentences = []
    for sent in doc.sents:
        if any(word in sent.text.lower() for word in ["year", "experience", "worked", "internship", "project"]):
            exp_sentences.append(sent.text)
    return exp_sentences

def match_resume_to_jobs(resume_text, top_k=3):
    resume_vec = model.encode([resume_text])
    similarities = cosine_similarity(resume_vec, job_embeddings)[0]
    job_df["match_score"] = similarities
    top_jobs = job_df.sort_values(by="match_score", ascending=False).head(top_k)
    return top_jobs

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening & Matching System")
st.write("Upload your resume to get automatically matched with the most suitable job descriptions.")

uploaded_resume = st.file_uploader("📤 Upload Resume (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_resume:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_resume_text(uploaded_resume)
        skills = extract_skills(resume_text)
        experience = extract_experience(resume_text)

        st.subheader("🧠 Extracted Information")
        st.write(f"**Detected Skills:** {', '.join(skills) if skills else 'No skills detected'}")
        st.write("**Experience Mentions:**")
        for sent in experience[:3]:
            st.write(f"- {sent}")

        st.subheader("🔍 Matching Results")
        top_jobs = match_resume_to_jobs(resume_text, top_k=3)

        for _, row in top_jobs.iterrows():
            st.markdown(f"""
            **🎯 Job Title:** {row['Job Title']}
            **🏢 Company:** {row['Company']}
            **📊 Match Score:** {round(row['match_score']*100, 2)}%
            **🧩 Required Skills:** {row['skills']}
            ---
            """)

else:
    st.info("Please upload a resume to start screening.")

!jupyter nbconvert --to script "/content/drive/MyDrive/Colab Notebooks/Resume_Screening_app/app.ipynb"

!mv "/content/drive/MyDrive/Colab Notebooks/Resume_Screening_app/app.txt" "/content/drive/MyDrive/Colab Notebooks/Text_Summarization_app/app.py"
