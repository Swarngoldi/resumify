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

# Try to load a 3-class (neg/neu/pos) sentiment model first; fall back to 2-class.
print("Loading sentiment model (preferring 3-class)...")
try:
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    SENTIMENT_MODE = "multi"  # negative / neutral / positive (sentence-level aggregation)
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
    allow_origins=["*"],           # tighten in production
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

def _map_field_to_category(s: str) -> str:
    s = (s or "").lower()
    for k in RECOMMENDATIONS_DB.keys():
        if k.lower() in s:
            return k
    if "full stack" in s or "software" in s or "web" in s or "frontend" in s or "backend" in s:
        return "Web Development"
    if "data" in s or "ml" in s or "machine learning" in s or "ai" in s:
        return "Data Science"
    if "devops" in s or "sre" in s or "site reliability" in s:
        return "DevOps"
    if "android" in s or "kotlin" in s:
        return "Android Development"
    if "ios" in s or "swift" in s:
        return "iOS Development"
    if "ui" in s or "ux" in s or "design" in s or "figma" in s:
        return "UI/UX Design"
    return "default"

# ---------- UTILITIES ----------

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def _regex_email(text: str) -> Optional[str]:
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else None

def _regex_phone(text: str) -> Optional[str]:
    m = re.search(r'(\(?\+?\d{1,3}\)?[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})', text)
    return m.group(0) if m else None

def extract_name_spacy(text: str) -> Optional[str]:
    # First 1200 chars: likely header area
    doc = nlp(text[:1200])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return None

def clean_tokens(s: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9+#.]+", (s or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]

def build_phrase_matcher(skills_map: Dict[str, List[str]]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    seen = set()
    for field, skills in skills_map.items():
        patterns = []
        for s in skills:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            patterns.append(nlp.make_doc(s))
        if patterns:
            matcher.add(field, patterns)
    return matcher

PHRASE_MATCHER = build_phrase_matcher(SKILLS_DB)

def extract_skills_spacy(text: str) -> Tuple[Dict[str, List[str]], List[str]]:
    doc = nlp(text)
    by_field: Dict[str, List[str]] = {k: [] for k in SKILLS_DB}
    found = set()
    matches = PHRASE_MATCHER(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text.lower()
        if span == "node":
            span = "nodejs"
        if span == "nextjs":
            span = "next.js"
        if span == "sklearn":
            span = "scikit-learn"
        if span not in found:
            by_field[label].append(span)
            found.add(span)
    toks = set(clean_tokens(text))
    for field, skills in SKILLS_DB.items():
        for s in skills:
            if " " not in s and s.lower() in toks:
                if s.lower() not in found:
                    by_field[field].append(s.lower())
                    found.add(s.lower())
    all_skills = sorted(found)
    return by_field, all_skills

DEGREE_PATTERNS = [
    r"(b\.?tech|b\.?e\.?|bachelor(?:'s)?(?: of)? (?:technology|engineering|science|arts))",
    r"(m\.?tech|m\.?e\.?|master(?:'s)?(?: of)? (?:technology|engineering|science|arts|computer applications))",
    r"(bsc|msc|bca|mca|mba|ph\.?d\.?)"
]

def extract_education(text: str) -> List[str]:
    edu = []
    for pat in DEGREE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            snippet = " ".join(text[start:end].split())
            if snippet not in edu:
                edu.append(snippet)
    return edu[:6]

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")

def estimate_years_experience(text: str) -> int:
    years = YEAR_PATTERN.findall(text)
    ranges = re.findall(r"((19|20)\d{2})\s*[-–—]\s*((19|20)\d{2}|present|current)", text, flags=re.IGNORECASE)
    total = 0
    spans = []
    for rng in ranges:
        try:
            start = int(rng[0])
            end_raw = rng[2]
            if re.match(r"(?i)present|current", end_raw):
                from datetime import datetime
                end = datetime.now().year
            else:
                end = int(end_raw)
            if end >= start and (end - start) <= 50:
                spans.append((start, end))
        except Exception:
            pass
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    for s, e in merged:
        total += (e - s)
    for m in re.finditer(r"(\d+)\s*\+?\s*years?", text, flags=re.IGNORECASE):
        try:
            val = int(m.group(1))
            total = max(total, val)
        except:
            pass
    return max(0, min(total, 40))

def predict_experience_level(text: str) -> str:
    yrs = estimate_years_experience(text)
    if yrs >= 3:
        return "Experienced"
    if yrs >= 1:
        return "Intermediate"
    tl = text.lower()
    if 'internship' in tl or 'intern' in tl:
        return "Intermediate"
    return "Fresher"

def skills_from_job_desc(job_description: str) -> List[str]:
    _, all_sk = extract_skills_spacy(job_description)
    return all_sk

def predicted_field_from_skills(resume_skills: List[str]) -> str:
    best_field, best_hits = "default", 0
    rs = set(resume_skills)
    for field, skills in SKILLS_DB.items():
        hits = sum(1 for s in skills if s.lower() in rs)
        if hits > best_hits:
            best_hits = hits
            best_field = field
    return best_field

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def local_match_score(resume_text: str, job_description: str, resume_skills: List[str], jd_skills: List[str]) -> Tuple[int, Dict[str, float]]:
    toks_resume = clean_tokens(resume_text)
    toks_jd = clean_tokens(job_description)
    token_sim = jaccard(toks_resume, toks_jd)
    if jd_skills:
        skill_overlap = len(set(resume_skills) & set(jd_skills)) / max(1, len(set(jd_skills)))
    else:
        skill_overlap = 0.0
    score = round((0.7 * skill_overlap + 0.3 * token_sim) * 100)
    score = max(0, min(100, score))
    return score, {"skill_overlap": skill_overlap, "token_similarity": token_sim}

def extract_basic_info(text: str) -> Dict[str, Optional[str]]:
    name = extract_name_spacy(text) or ""
    email = _regex_email(text)
    phone = _regex_phone(text)
    return {"name": name, "email": email, "mobile": phone}

# ---------- SENTIMENT HELPERS ----------

def compute_tone(text: str) -> Dict[str, object]:
    """
    Use a 3-class model at sentence-level if available; otherwise fall back to 2-class doc-level.
    Returns: {label, score, distribution}
    """
    if SENTIMENT_MODE == "multi":
        doc = nlp(text[:6000])
        sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 3][:30]
        if not sents:
            return {"label": "NEUTRAL", "score": 50, "distribution": {"NEGATIVE": 0, "NEUTRAL": 100, "POSITIVE": 0}}
        preds = sentiment_model(sents, truncation=True)
        # Map labels robustly (some versions return LABEL_0/1/2)
        label_map = {
            "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
            "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"
        }
        counts = Counter(label_map.get(p["label"], p["label"]).upper() for p in preds)
        for k in ("NEGATIVE", "NEUTRAL", "POSITIVE"):
            counts.setdefault(k, 0)
        total = sum(counts.values()) or 1
        dist = {k: round(v * 100 / total) for k, v in counts.items()}
        label = max(dist, key=dist.get)
        return {"label": label, "score": dist[label], "distribution": dist}
    else:
        res = sentiment_model(text[:2000])
        label = res[0]['label'].upper()
        score = int(res[0]['score'] * 100)
        if label not in ("NEGATIVE", "POSITIVE"):
            label = "POSITIVE" if "POS" in label else "NEGATIVE"
        dist = {
            "NEGATIVE": (100 - score) if label == "POSITIVE" else score,
            "NEUTRAL": 0,
            "POSITIVE": score if label == "POSITIVE" else (100 - score)
        }
        return {"label": label, "score": score, "distribution": dist}

# ---------- API ----------

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

    # ---------- NLP EXTRACT ----------
    basic_info = extract_basic_info(resume_text)
    years_exp = estimate_years_experience(resume_text)
    experience_level = predict_experience_level(resume_text)
    edu = extract_education(resume_text)
    by_field, resume_skills = extract_skills_spacy(resume_text)
    jd_skills = skills_from_job_desc(job_description)
    missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))
    predicted_field_local = predicted_field_from_skills(resume_skills)
    match_score_local, breakdown = local_match_score(resume_text, job_description, resume_skills, jd_skills)

    # Sentiment (improved)
    tone_info = compute_tone(resume_text)
    tone_label = tone_info["label"]
    tone_score = tone_info["score"]
    tone_dist = tone_info["distribution"]

    # ---------- LLM CALL ----------
    try:
        api_key = "AIzaSyBHjcZdtfa52RILdP8mYWrjBeWcmP7P030"  # move to env var in prod
        prompt = f"""
Act as an expert technical recruiter.
You will get:
- Resume text
- Job description
- NLP-extracted skills, education, estimated years of experience, and a local match score.

Return ONLY a valid JSON with keys:
"jobMatchScore","scoreReasoning","predictedField","strengths","weaknesses","summary","interviewQuestions".

Context (from NLP):
- Estimated Years Experience: {years_exp}
- Experience Level: {experience_level}
- Resume Skills: {resume_skills}
- Job Skills: {jd_skills}
- Missing Skills: {missing_skills}
- Local Match Score (0-100): {match_score_local}
- Resume Tone: {tone_label} (Confidence: {tone_score}% | Dist: {tone_dist})
- Education snippets: {edu[:3]}

Job Description:
---
{job_description if job_description.strip() else "No job description provided. Analyze the resume for general software roles."}
---

Resume Text:
---
{resume_text}
---
"""
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if result.get("candidates"):
                content_string = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                raise HTTPException(status_code=500, detail="Invalid response from Gemini API.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    # ---------- RESPONSE ----------
    return {
        "basicInfo": basic_info,
        "experienceLevel": experience_level,
        "tone": {
            "label": tone_label,
            "score": tone_score,
            "distribution": tone_dist,
            "note": "Sentence-level sentiment using a 3-class model when available."
        },
        "aiAnalysis": content_string,  # UI expects a JSON string here
        "nlp": {
            "yearsExperience": years_exp,
            "education": edu,
            "skillsByField": by_field,
            "skillsAll": resume_skills,
            "jobSkills": jd_skills,
            "missingSkills": missing_skills,
            "predictedFieldLocal": predicted_field_local,
            "localMatchScore": match_score_local,
            "scoreBreakdown": breakdown
        }
    }

@app.get("/recommendations/{field:path}")
async def get_recommendations(field: str):
    field = unquote(field)
    cat = _map_field_to_category(field)
    field_recs = RECOMMENDATIONS_DB.get(cat, RECOMMENDATIONS_DB["default"])
    return {
        "courses": random.sample(field_recs.get("courses", []), k=min(2, len(field_recs.get("courses", [])))),
        "resume_videos": random.sample(field_recs.get("resume_videos", RECOMMENDATIONS_DB["default"]["resume_videos"]), 1),
        "interview_videos": random.sample(field_recs.get("interview_videos", RECOMMENDATIONS_DB["default"]["interview_videos"]), 1)
    }

# ---------- RUN ----------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
