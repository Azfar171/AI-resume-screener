# 📄 AI Resume Screener

An end-to-end **AI-powered Resume Screening** web app built with **Streamlit**, using NLP techniques — TF-IDF vectorisation and Cosine Similarity — to intelligently rank candidates against a job description.

---

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Feature | Description |
|---|---|
| NLP Preprocessing | Tokenisation · Stopword removal · Lemmatisation |
| TF-IDF Vectorisation | Bi-gram support · Sublinear TF scaling |
| Cosine Similarity Ranking | Compare any number of resumes against a JD |
| Interactive Threshold | Tune the shortlist cut-off in real time |
| Visualisations | Match bar chart + TF-IDF weight comparison |
| Export-ready table | Raw scores table for download |

---

## 🛠 Local Setup

```bash
# 1. Clone / copy the project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → *New app*.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done! 🎉

---

## 📁 Project Structure

```
Resume cleaning/
├── app.py              ← Streamlit frontend (this is the entry point)
├── main.py             ← FastAPI backend (optional REST API)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── AI_Resume_Screening.ipynb  ← Original research notebook
```

---

## 🧠 How It Works

1. **Preprocessing** – Text is lowercased, cleaned, tokenised, stopwords removed, and lemmatised.
2. **TF-IDF** – All documents (JD + resumes) are vectorised with a shared vocabulary (max 3 000 features, 1–2 n-grams).
3. **Cosine Similarity** – Each resume vector is compared to the JD vector; the resulting score (0–1) reflects semantic overlap.
4. **Ranking** – Candidates are sorted by score; those above the configurable threshold are shortlisted.

---

## 🔧 FastAPI Backend (Optional)

`main.py` exposes a REST API:

```bash
pip install fastapi uvicorn python-multipart
uvicorn main:app --reload
# POST /rank  →  { "job_description": "...", "candidates": [...] }
```

---


