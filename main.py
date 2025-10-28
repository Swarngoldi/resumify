# main.py - FastAPI Backend (AI Resume Analyzer with NLP upgrades; no job/auto-apply endpoints)

import os
import re
import uvicorn
import httpx
import json
from typing import Optional, List, Dict, Tuple
from collections import Counter
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from spacy.matcher import PhraseMatcher
import tempfile
import nltk
from nltk.corpus import stopwords
import traceback
from PyPDF2 import PdfReader
import random
from transformers import pipeline  # Sentiment (HF)
from urllib.parse import unquote

# ---------- MODELS & DATA ----------

print("Loading sentiment model (preferring 3-class)...")
try:
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    SENTIMENT_MODE = "multi"
    print("Loaded 3-class sentiment model.")
except Exception as e:
    print("Falling back to 2-class sentiment model:", e)
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    SENTIMENT_MODE = "binary"
    print("Loaded 2-class sentiment model.")

# Ensure NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords'...")
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

# Ensure spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(
    title="Next-Gen AI Resume Analyzer API",
    description="NLP-enhanced resume analyzer with LLM reasoning.",
    version="7.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DOMAIN DATA ----------

SKILLS_DB: Dict[str, List[str]] = {
    'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'nodejs', 'node', 'express', 'django', 'flask', 'php', 'ruby', 'api', 'rest', 'typescript', 'tailwind', 'mongodb', 'postgres', 'mysql', 'next.js', 'nextjs', 'mern'],
    'Data Science': ['python', 'r', 'sql', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'scikit-learn', 'sklearn', 'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'nlp', 'statistics', 'tableau', 'powerbi', 'beautifulsoup'],
    'Android Development': ['java', 'kotlin', 'android', 'android sdk', 'xml', 'jetpack', 'firebase', 'rxjava', 'dagger'],
    'iOS Development': ['swift', 'objective-c', 'xcode', 'uikit', 'swiftui', 'coredata', 'ios'],
    'DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'ci/cd', 'git', 'linux', 'prometheus', 'grafana'],
    'UI/UX Design': ['ux', 'ui', 'figma', 'sketch', 'adobe xd', 'wireframing', 'prototyping', 'user research', 'design thinking', 'shadcn']
}

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

# (Keep all functions exactly as in your version — extract_text_from_pdf, extract_skills_spacy, compute_tone, analyze_resume_with_ai, etc.)

@app.get("/recommendations/{field:path}")
async def get_recommendations(field: str):
    field = unquote(field)
    cat = _map_field_to_category(field)
    field_recs = RECOMMENDATIONS_DB.get(cat, RECOMMENDATIONS_DB["default"])
    return {
        "courses": random.sample(field_recs.get("courses", []), k=min(2, len(field_recs.get("courses", [])))),
        "resume_videos": field_recs.get("resume_videos", []),
        "interview_videos": field_recs.get("interview_videos", [])
    }
