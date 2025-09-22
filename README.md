# Next‑Gen AI Resume Analyzer

An end‑to‑end, local-first resume screening tool. Upload a **PDF resume**, optionally paste a **job description**, and get an **AI job‑match score**, strengths/weaknesses, a concise summary, and curated learning/interview resources. The project consists of a static **frontend** (`index.html`) and a **FastAPI backend** (`main.py`).

---

## ✨ Features

- **Drag‑and‑drop PDF upload** with instant validation.  
- **Optional job description** input to tailor the analysis.  
- **AI Insights**: job‑match score, reasoning, strengths, weaknesses, and summary.  
- **Basic info extraction** (name, email, experience level).  
- **Tone analysis** of the resume using a Transformer pipeline.  
- **Personalized recommendations** (courses + videos) based on predicted field.  
- **CORS‑enabled** API for easy local dev and deployment.  

---

## 🧱 Architecture (High‑Level)

```
[Browser UI]
 index.html (Tailwind + Vanilla JS)
       |  fetch() multipart/form-data
       v
[FastAPI Backend]
 /analyze        -> parse PDF (PyPDF2)
                 -> extract entities (regex + heuristics)
                 -> sentiment (Transformers pipeline)
                 -> LLM call (Gemini) for structured insights
                 -> JSON payload back to client

 /recommendations/{field} -> returns curated links
```

---

## 🧰 Tech Stack

**Frontend**: HTML, TailwindCSS (CDN), Vanilla JS  
**Backend**: FastAPI, Uvicorn, httpx, PyPDF2, spaCy (en_core_web_sm), NLTK, Hugging Face `transformers` (DistilBERT sentiment pipeline)  
**LLM**: Google Gemini (server‑side call)  
**Language**: Python 3.9+ (recommended 3.10/3.11)

---

## 🚀 Quickstart (Local)

### 1) Clone & enter the project
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -U pip wheel
pip install fastapi "uvicorn[standard]" httpx pypdf2 nltk spacy transformers torch
# Optional: if torch fails, try: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> The backend will auto‑download NLTK stopwords and the **spaCy** model `en_core_web_sm` on first run if missing.

### 4) Set your API key (required for LLM)
Create a `.env` or export an env var (preferred):
```bash
# macOS/Linux
export GOOGLE_API_KEY="YOUR_GEMINI_KEY"
# Windows (PowerShell)
$env:GOOGLE_API_KEY="YOUR_GEMINI_KEY"
```

> **Important:** The starter code currently contains a hard‑coded `api_key` string. Replace it with `os.getenv("GOOGLE_API_KEY")` before deploying. See **Security Notes** below.

### 5) Run the backend
```bash
uvicorn main:app --reload --port 8000
```

### 6) Open the frontend
- Option A: Double‑click `index.html` to open in your browser.
- Option B (recommended): Serve it from a tiny local server so relative links work consistently:
  ```bash
  python -m http.server 5500
  # open http://127.0.0.1:5500/index.html
  ```

The frontend is already configured to call `http://127.0.0.1:8000` for API requests.

---

## 🔌 API Reference

### POST `/analyze/`  
Analyze a resume (PDF) against an optional job description.

**Request (multipart/form-data):**
- `file`: *PDF* resume (required)
- `job_description`: string (optional)

**Curl example:**
```bash
curl -X POST http://127.0.0.1:8000/analyze/ \
  -F "file=@/path/to/resume.pdf" \
  -F "job_description=Software Engineer with React + FastAPI"
```

**Response (JSON):**
```json
{
  "basicInfo": {
    "name": "Jane Doe",
    "email": "jane@doe.com",
    "mobile": "+1 555 123 4567"
  },
  "experienceLevel": "Fresher | Intermediate | Experienced",
  "tone": { "label": "POSITIVE", "score": 97, "note": "This is resume sentiment, not job match." },
  "aiAnalysis": "{ ... stringified JSON with jobMatchScore, predictedField, strengths, weaknesses, summary, interviewQuestions ... }"
}
```
> Note: `aiAnalysis` is **a string containing JSON** (from the LLM). The frontend parses it before rendering.

---

### GET `/recommendations/{field}`  
Return curated courses and videos for a predicted field (e.g., `"Data Science"`, `"Web Development"`, etc.).

**Example:**
```bash
curl http://127.0.0.1:8000/recommendations/Web%20Development
```
**Response (JSON):**
```json
{
  "courses": [["Course Title", "https://link"]],
  "resume_videos": ["https://youtube-link"],
  "interview_videos": ["https://youtube-link"]
}
```

---

## 📁 Project Structure

```
.
├── index.html      # Frontend (Tailwind + Vanilla JS, drag-drop, progress ring, insights)
├── main.py         # FastAPI backend with PDF parsing, sentiment + Gemini integration
└── README.md       # You are here
```

---

## 🔐 Security Notes

- **Never commit API keys**. Read from env (`os.getenv("GOOGLE_API_KEY")`) instead of hard‑coding.  
- Scope your CORS `allow_origins` to known hosts for production.  
- Treat uploaded PDFs as **untrusted**: the code writes to a temp file and deletes it, but consider scanning/size limits.  
- The LLM call runs server‑side; do not expose secrets to the client.  
- Check your Gemini account’s **quotas/billing** and add retries/backoff for robustness.

---

## 🧪 Troubleshooting

- **“Error processing PDF”** → Ensure the resume is a *text‑based* PDF (not image‑only). Use OCR if needed.  
- **“Invalid response from Gemini API”** → Verify `GOOGLE_API_KEY`, network access, and quota.  
- **Transformer errors (torch/tensorflow)** → Install `torch` (CPU build) or TensorFlow; the DistilBERT pipeline can run on CPU.  
- **CORS issues** → In production, set `allow_origins=["https://your-frontend.app"]`.  
- **Slow first run** → Model downloads on first use; subsequent runs are faster.

---

## 📦 Deployment Tips

- **Backend**: Deploy on Render/Railway/Fly.io/AWS. Expose port `8000` (or behind Nginx). Set `GOOGLE_API_KEY` in env.  
- **Frontend**: Host `index.html` on Netlify/Vercel/GitHub Pages; point it to your backend base URL.  
- **HTTPS**: Use HTTPS in production and update the frontend fetch URL accordingly.

---

## 🙌 Acknowledgements

- Hugging Face `transformers` (DistilBERT sentiment)  
- spaCy `en_core_web_sm` model  
- Google Gemini for LLM analysis

---

## 📄 License

MIT (or your preferred license). Update this section before publishing.
