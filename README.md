# resumify
# Next-Gen AI Resume Analyzer

An end-to-end, local-first resume screening tool. Upload a **PDF resume**, optionally paste a **job description**, and get an **AI job-match score**, strengths/weaknesses, a concise summary, and curated learning/interview resources. The project consists of a static **frontend** (`index.html`) and a **FastAPI backend** (`main.py`).

---

## ✨ Features

* **Drag-and-drop PDF upload** with instant validation
* **Optional job description** input to tailor the analysis
* **AI insights**: job-match score, reasoning, strengths, weaknesses, summary
* **Basic info extraction** (name, email, phone) + rough experience signal
* **(Optional)** tone/sentiment analysis on the resume text
* **Personalized recommendations** (courses + YouTube resources) by predicted field
* **CORS-enabled** API; clean separation of frontend & backend

---

## 🧱 Architecture (High-Level)

```
[Browser UI]
 index.html (Tailwind + Vanilla JS)
       |  fetch() multipart/form-data
       v
[FastAPI Backend]
 /analyze               -> parse PDF (e.g., PyPDF2)
                        -> extract entities (regex/heuristics)
                        -> optional sentiment (Transformers)
                        -> LLM (e.g., Gemini) for structured insights
                        -> JSON back to client

 /recommendations/{field} -> returns curated links (courses/videos)
```

---

## 🧰 Tech Stack

**Frontend:** HTML, TailwindCSS (CDN), Vanilla JS
**Backend:** FastAPI, Uvicorn, PyPDF2 (PDF parsing), optional: spaCy / NLTK, Hugging Face `transformers`
**LLM:** Google Gemini (server-side)
**Language:** Python 3.9+ (recommend 3.10/3.11)

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
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -U pip wheel
pip install fastapi "uvicorn[standard]" httpx pypdf2
# Optional extras if your backend uses them:
pip install nltk spacy transformers torch
# If torch CPU wheel is needed:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4) Configure environment variables

Create a `.env` or export in shell:

```bash
# macOS/Linux
export GOOGLE_API_KEY="YOUR_GEMINI_KEY"
# Windows (PowerShell)
$env:GOOGLE_API_KEY="YOUR_GEMINI_KEY"
```

> If your current `main.py` has a hard-coded key, replace it with `os.getenv("GOOGLE_API_KEY")` before deploying.

### 5) Run the backend

```bash
uvicorn main:app --reload --port 8000
```

### 6) Open the frontend

* **Option A:** Double-click `index.html` to open in your browser
* **Option B (recommended):** Serve it locally for consistent relative paths

  ```bash
  python -m http.server 5500
  # open http://127.0.0.1:5500/index.html
  ```

The frontend calls `http://127.0.0.1:8000` by default—update it if you change ports/hosts.

---

## 🔌 API Reference

### POST `/analyze/` — Analyze a resume

**Request (multipart/form-data):**

* `file`: PDF resume (required)
* `job_description`: string (optional)

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
  "tone": { "label": "POSITIVE", "score": 97, "note": "Sentiment of resume text" },
  "aiAnalysis": "{ ...stringified JSON with jobMatchScore, predictedField, strengths, weaknesses, summary, interviewQuestions ... }"
}
```

> Note: `aiAnalysis` is a **string containing JSON** returned by the LLM; the frontend parses it before rendering.

---

### GET `/recommendations/{field}` — Curated resources

Returns courses and videos for the predicted/selected field.

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
├── index.html      # Frontend (Tailwind + Vanilla JS)
├── main.py         # FastAPI backend (PDF parsing + LLM integration)
└── README.md       # This file
```

---

## ⚙️ Environment Variables

* `GOOGLE_API_KEY` — API key for Gemini (server-side only)
* `PORT` (optional) — override Uvicorn port (default 8000)
* Any additional keys for third-party services you add later

---

## 🔐 Security Notes

* **Do not commit secrets**. Use environment variables or a secrets manager.
* Restrict CORS in production to your frontend origin.
* Treat uploaded PDFs as **untrusted**; consider file size limits and malware scanning.
* Never expose LLM keys in the browser; all calls happen **server-side**.
* Add basic **rate limiting** and **request size limits** for public deployments.

---

## 🧪 Troubleshooting

* **“Error parsing PDF”** → Ensure the resume is text-based; OCR image-only PDFs if needed.
* **“Invalid LLM response / 401”** → Check `GOOGLE_API_KEY`, billing, and network access.
* **Transformer model download slow** → First run may download models; subsequent runs are faster.
* **CORS errors** → In production set `allow_origins=["https://your-frontend.app"]`.
* **Windows path issues** → Use escaped paths or WSL for shell commands.

---

## 📦 Deployment

* **Backend**: Render, Railway, Fly.io, AWS, GCP, Azure. Run with:

  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```

  Put `GOOGLE_API_KEY` in environment settings.

* **Frontend**: Netlify, Vercel, GitHub Pages, or any static host.
  Update the frontend’s API base URL to your deployed backend.

* **HTTPS**: Use HTTPS in production and update any hard-coded `http://` URLs.

---

## 🗺️ Roadmap Ideas

* Real-time **skills gap** highlighting against a JD
* **Job scraper** integration for LinkedIn/Indeed APIs or RSS (respect ToS)
* **Versioned scoring history** for multiple resume iterations
* One-click **PDF report** export with recommendations
* **Auth** + user profiles to save analyses

---

## 🙌 Acknowledgements

* Hugging Face `transformers`
* spaCy `en_core_web_sm`
* Google Gemini (Generative AI)


---

### Contributing

PRs and issues welcome! Please open an issue with a clear description and steps to reproduce.

