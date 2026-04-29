
# backend/main.py run with: uvicorn main:app --reload
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

app = FastAPI(title="Resume Screening API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STOP = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [LEM.lemmatize(t) for t in tokens if t not in STOP and len(t) > 2]
    return " ".join(tokens)

class RankRequest(BaseModel):
    job_description: str
    candidates: List[dict]   # [{"name": str, "resume": str}, ...]

class RankResult(BaseModel):
    rank: int
    name: str
    score: float
    match_pct: float

@app.get("/")
def root():
    return {"status": "ok", "message": "Resume Screening API"}

@app.post("/rank", response_model=List[RankResult])
def rank_candidates(req: RankRequest):
    jd_proc = preprocess(req.job_description)
    res_proc = [preprocess(c["resume"]) for c in req.candidates]
    docs = [jd_proc] + res_proc

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3000)
    mat = vec.fit_transform(docs)
    scores = cosine_similarity(mat[0], mat[1:])[0]

    max_s = max(scores) if max(scores) > 0 else 1
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    return [
        RankResult(
            rank=i + 1,
            name=req.candidates[idx]["name"],
            score=round(float(s), 4),
            match_pct=round(float(s / max_s) * 100, 1)
        )
        for i, (idx, s) in enumerate(ranked)
    ]
