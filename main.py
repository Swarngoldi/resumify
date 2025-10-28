# main.py - The FastAPI Backend (Final Version with TensorFlow & All Features)

import os
import re
import uvicorn
import httpx
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import spacy
import tempfile
import nltk
import traceback
from PyPDF2 import PdfReader
import random
from transformers import pipeline # For TensorFlow Sentiment Analysis

# --- DATA & MODEL LOADING ---

# Load Sentiment Analysis model once on startup
print("Loading TensorFlow sentiment analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("Model loaded successfully.")

# --- Download necessary NLTK data if not already present ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' for NLTK...")
    nltk.download('stopwords')

# --- Download and load SpaCy model ---
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading 'en_core_web_sm' for SpaCy...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Next-Gen AI Resume Analyzer API",
    description="An advanced API that uses a Large Language Model and TensorFlow to analyze resumes.",
    version="5.1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA DEFINITIONS (Expanded) ---
SKILLS_DB = {
    'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'php', 'ruby', 'api', 'typescript', 'tailwind', 'mongodb', 'rest', 'next.js', 'mern'],
    'Data Science': ['python', 'r', 'sql', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'statistics', 'nlp', 'tableau', 'beautifulsoup'],
    'Android Development': ['java', 'kotlin', 'android sdk', 'xml', 'jetpack', 'firebase', 'rxjava', 'dagger'],
    'iOS Development': ['swift', 'objective-c', 'xcode', 'uikit', 'swiftui', 'coredata'],
    'DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'ci/cd', 'git'],
    'UI/UX Design': ['ux', 'ui', 'figma', 'sketch', 'adobe xd', 'wireframing', 'prototyping', 'user research', 'design thinking', 'shadcn']
}

# --- RECOMMENDATIONS DB with field-specific videos ---
RECOMMENDATIONS_DB = {
    "Data Science": {
        "courses": [
            ["Machine Learning Crash Course by Google", "https://developers.google.com/machine-learning/crash-course"],
            ["Data Scientist with Python by DataCamp", "https://www.datacamp.com/tracks/data-scientist-with-python"]
        ],
        "resume_videos": ["https://youtu.be/GwIo3gDZCVQ"],
        "interview_videos": ["https://youtu.be/H1NQnxZ3aD8"]
    },
    "Web Development": {
        "courses": [
            ["Python and Django Full Stack Web Developer Bootcamp", "https://www.udemy.com/course/python-and-django-full-stack-web-developer-bootcamp"],
            ["React Crash Course [Free]", "https://youtu.be/Dorf8i6lCuk"]
        ],
        "resume_videos": ["https://youtu.be/1Rs2ND1ryYc"],
        "interview_videos": ["https://youtu.be/3Q_oYDQ2whs"]
    },
    "Android Development": {
        "courses": [
            ["Android App Development Specialization", "https://www.coursera.org/specializations/android-app-development"],
            ["Become an Android Kotlin Developer", "https://www.udacity.com/course/android-kotlin-developer-nanodegree--nd940"]
        ],
        "resume_videos": ["https://youtu.be/EFK6eR5jYtM"],
        "interview_videos": ["https://youtu.be/LsM8WZxV7n4"]
    },
    "iOS Development": {
        "courses": [
            ["iOS App Development with Swift", "https://www.coursera.org/learn/ios-app-development-swift"],
            ["SwiftUI Essentials", "https://developer.apple.com/tutorials/swiftui"]
        ],
        "resume_videos": ["https://youtu.be/n5X_V81OYnQ"],
        "interview_videos": ["https://youtu.be/mI5Bq6gso8s"]
    },
    "DevOps": {
        "courses": [
            ["Docker & Kubernetes: The Practical Guide", "https://www.udemy.com/course/docker-kubernetes-the-practical-guide/"],
            ["AWS Certified DevOps Engineer", "https://aws.amazon.com/certification/certified-devops-engineer-professional/"]
        ],
        "resume_videos": ["https://youtu.be/5p8wTOr8AbU"],
        "interview_videos": ["https://youtu.be/2Tofun1W48Q"]
    },
    "UI/UX Design": {
        "courses": [
            ["Google UX Design Certificate", "https://www.coursera.org/professional-certificates/google-ux-design"],
            ["Figma for Beginners", "https://help.figma.com/hc/en-us/articles/360040518034-Get-started-in-Figma"]
        ],
        "resume_videos": ["https://youtu.be/u75hUSShvnc"],
        "interview_videos": ["https://youtu.be/4XgY1eK69AI"]
    },
    "default": {
        "resume_videos": ["https://youtu.be/y8YH0Qbu5h4"],
        "interview_videos": ["https://youtu.be/HG68Ymazo18"]
    }
}

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_basic_info(text):
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.search(r'(\(?\+?\d{1,3}\)?[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})', text)
    
    name = ""
    for line in text.split('\n'):
        if line.strip():
            name = line.strip()
            break

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "mobile": phone.group(0) if phone else None
    }

def predict_experience_level(text):
    text_lower = text.lower()
    if 'experience' in text_lower or 'work history' in text_lower:
        return "Experienced"
    if 'internship' in text_lower or 'intern' in text_lower:
        return "Intermediate"
    return "Fresher"

# --- API ENDPOINTS ---

@app.post("/analyze/")
async def analyze_resume_with_ai(file: UploadFile = File(...), job_description: str = Form("")):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        resume_text = extract_text_from_pdf(tmp_path)
        if not resume_text.strip():
            raise ValueError("Could not extract any text from the PDF.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    basic_info = extract_basic_info(resume_text)
    experience_level = predict_experience_level(resume_text)
    
    sentiment_result = sentiment_analyzer(resume_text[:512])
    tone_label = sentiment_result[0]['label']
    tone_score = int(sentiment_result[0]['score'] * 100)

    try:
        api_key = "your_api_key"
        prompt = f"""
        Act as an expert technical recruiter. Analyze the following resume against the provided job description, considering the candidate's predicted experience level and the sentiment of their resume. 
        Provide a JSON response with: jobMatchScore, scoreReasoning, predictedField, strengths, weaknesses, summary, interviewQuestions.

        CONTEXT:
        - Predicted Experience Level: {experience_level}
        - Resume Tone: {tone_label} (Confidence: {tone_score}%)

        Job Description:
        ---
        {job_description if job_description.strip() else "No job description provided. Analyze the resume for general strength as a software developer."}
        ---

        Resume Text:
        ---
        {resume_text}
        ---
        """

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result.get('candidates'):
                content_string = result['candidates'][0]['content']['parts'][0]['text']
                return {
                    "basicInfo": basic_info,
                    "experienceLevel": experience_level,
                    "tone": {"label": tone_label, "score": tone_score, "note": "This is resume sentiment, not job match."},
                    "aiAnalysis": content_string
                }
            else:
                raise HTTPException(status_code=500, detail="Invalid response from Gemini API.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/recommendations/{field}")
async def get_recommendations(field: str):
    field_recs = RECOMMENDATIONS_DB.get(field, RECOMMENDATIONS_DB["default"])
    return {
        "courses": random.sample(field_recs.get("courses", []), k=min(2, len(field_recs.get("courses", [])))),
        "resume_videos": random.sample(field_recs.get("resume_videos", RECOMMENDATIONS_DB["default"]["resume_videos"]), 1),
        "interview_videos": random.sample(field_recs.get("interview_videos", RECOMMENDATIONS_DB["default"]["interview_videos"]), 1)
    }
