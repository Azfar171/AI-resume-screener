import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from typing import List, Dict, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLTK Downloads ───────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* Rank badge */
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 50%;
    width: 2rem;
    height: 2rem;
    line-height: 2rem;
    text-align: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 0.5rem;
}

/* Match bar */
.match-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 99px;
    height: 10px;
    margin-top: 6px;
}
.match-bar-fill {
    border-radius: 99px;
    height: 10px;
    background: linear-gradient(90deg, #6366f1, #34d399);
}

/* Section heading */
.section-heading {
    color: #e2e8f0;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    border-left: 4px solid #6366f1;
    padding-left: 0.6rem;
}

/* Shortlisted / rejected tags */
.tag-shortlist {
    background: rgba(52, 211, 153, 0.2);
    color: #34d399;
    border: 1px solid #34d399;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.tag-reject {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid #f87171;
    border-radius: 8px;
    padding: 0.2rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 0.4rem;
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.7) !important;
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* Scrollable keyword list */
.kw-chip {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    color: #a5b4fc;
    border-radius: 99px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    margin: 2px 3px;
    border: 1px solid rgba(99,102,241,0.4);
}

/* Dataframe tweaks */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── NLP Classes ──────────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def normalize(self, tokens: List[str]) -> List[str]:
        if self.lemmatizer:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    def process(self, text: str) -> str:
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.normalize(tokens)
        return " ".join(tokens)

    def extract_features(self, text: str) -> Dict:
        tokens = self.tokenize(self.clean(text))
        content = self.remove_stopwords(tokens)
        return {
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "content_tokens": len(content),
            "unique_content": len(set(content)),
            "lexical_diversity": round(len(set(content)) / max(len(content), 1), 3),
            "top_terms": Counter(content).most_common(10),
        }


class TFIDFEngine:
    def __init__(self, ngram_range=(1, 2), max_features=3000, sublinear_tf=True):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            analyzer="word",
        )
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, documents: List[str]):
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self.tfidf_matrix

    def score(self, jd_vec, resume_vecs) -> np.ndarray:
        return cosine_similarity(jd_vec, resume_vecs)[0]

    def top_terms(self, vec_idx: int, n: int = 10) -> List[Tuple[str, float]]:
        vec = self.tfidf_matrix[vec_idx].toarray()[0]
        top_idx = vec.argsort()[::-1][:n]
        return [(self.feature_names[i], round(vec[i], 4)) for i in top_idx if vec[i] > 0]

    def term_weights(self, vec_idx: int, terms: List[str]) -> Dict[str, float]:
        """Return TF-IDF weight for each requested term in a given document row.
        Terms absent from the vocabulary get weight 0."""
        vec = self.tfidf_matrix[vec_idx].toarray()[0]
        feat_index = {f: i for i, f in enumerate(self.feature_names)}
        return {t: round(float(vec[feat_index[t]]), 4) if t in feat_index else 0.0
                for t in terms}


class CandidateRanker:
    def __init__(self, engine: TFIDFEngine, candidates: List[Dict]):
        self.engine = engine
        self.candidates = candidates

    def rank(self, jd_vec, resume_vecs) -> pd.DataFrame:
        raw_scores = self.engine.score(jd_vec, resume_vecs)
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1
        normalized = (raw_scores / max_score) * 100

        rows = []
        for i, (cand, raw, norm) in enumerate(zip(self.candidates, raw_scores, normalized)):
            top_kw = [t for t, _ in self.engine.top_terms(i + 1, 8)]
            rows.append({
                "Rank": 0,
                "Candidate": cand["name"],
                "Cosine Score": round(raw, 4),
                "Match %": round(norm, 1),
                "Top Keywords": ", ".join(top_kw[:5]),
            })

        df = pd.DataFrame(rows).sort_values("Cosine Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        return df[["Rank", "Candidate", "Cosine Score", "Match %", "Top Keywords"]]

    def decision(self, df: pd.DataFrame, threshold: float = 0.15) -> Dict:
        shortlisted = df[df["Cosine Score"] >= threshold]
        return {
            "shortlisted": shortlisted["Candidate"].tolist(),
            "rejected": df[df["Cosine Score"] < threshold]["Candidate"].tolist(),
            "threshold": threshold,
        }


# ─── Core pipeline ─────────────────────────────────────────────────────────────
def run_screening(job_description: str, resumes: List[Dict], threshold: float):
    preprocessor = TextPreprocessor()
    jd_processed = preprocessor.process(job_description)
    resumes_processed = [preprocessor.process(r["text"]) for r in resumes]
    jd_features = preprocessor.extract_features(job_description)

    all_docs = [jd_processed] + resumes_processed
    engine = TFIDFEngine(ngram_range=(1, 2), max_features=3000)
    tfidf_matrix = engine.fit_transform(all_docs)

    jd_vec = tfidf_matrix[0]
    resume_vecs = tfidf_matrix[1:]

    ranker = CandidateRanker(engine, resumes)
    results_df = ranker.rank(jd_vec, resume_vecs)
    decision = ranker.decision(results_df, threshold=threshold)

    return results_df, decision, jd_features, engine


# ─── Chart helpers ─────────────────────────────────────────────────────────────
def make_bar_chart(results_df: pd.DataFrame, threshold: float):
    fig, ax = plt.subplots(figsize=(8, max(3, len(results_df) * 0.65)))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#0f0c29")

    # Reverse so highest-ranked candidate appears at the top of the horizontal bar chart
    scores_rev = list(results_df["Cosine Score"][::-1])
    names_rev  = list(results_df["Candidate"][::-1])
    # Colors must match the same reversed order as the bars
    colors = ["#34d399" if s >= threshold else "#f87171" for s in scores_rev]
    bars = ax.barh(names_rev, scores_rev, color=colors, height=0.55)
    ax.axvline(threshold, color="#f59e0b", linestyle="--", linewidth=1.4, label=f"Threshold ({threshold})")

    ax.set_xlabel("Cosine Similarity Score", color="#94a3b8")
    ax.set_title("Candidate Match Scores", color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.12))   # matplotlib RGBA tuple, not CSS rgba()
    ax.legend(facecolor="#1e1b4b", edgecolor="none", labelcolor="#e2e8f0", fontsize=9)

    for bar, val in zip(bars, scores_rev):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", color="#e2e8f0", fontsize=8.5)

    plt.tight_layout()
    return fig


def make_tfidf_chart(results_df: pd.DataFrame, engine: TFIDFEngine, resumes: List[Dict]):
    # Find the top candidate's original position in the resumes list
    top_name = results_df.iloc[0]["Candidate"]
    top_cand_orig_idx = next(
        (i for i, r in enumerate(resumes) if r["name"] == top_name), 0
    )

    # ── Key fix: anchor x-axis to JD's top keywords ──────────────────────────
    # Previous version took the union of JD-top-10 ∪ candidate-top-10, so most
    # bars were 0 on one side (comparing completely different terms).
    # Now: pick the top 12 JD terms, then look up the candidate's weights for
    # those SAME terms → every bar is a true apples-to-apples comparison.
    jd_top = engine.top_terms(0, 12)                        # [(term, jd_weight), ...]
    terms  = [t for t, _ in jd_top]                         # x-axis labels
    jd_vals   = [w for _, w in jd_top]                      # JD weights
    cand_vals = list(engine.term_weights(top_cand_orig_idx + 1, terms).values())

    x     = np.arange(len(terms))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, max(4, len(terms) * 0.55)))
    fig.patch.set_facecolor("#0f0c29")
    ax.set_facecolor("#0f0c29")

    ax.bar(x - width / 2, jd_vals,   width, label="Job Description", color="#6366f1", alpha=0.85)
    ax.bar(x + width / 2, cand_vals, width, label=top_name,          color="#34d399", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=40, ha="right", color="#94a3b8", fontsize=8)
    ax.set_ylabel("TF-IDF Weight", color="#94a3b8")
    ax.set_title(
        f"Keyword Match: JD vs {top_name} (JD-anchored terms)",
        color="#e2e8f0", fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.12))
    ax.legend(facecolor="#1e1b4b", edgecolor="none", labelcolor="#e2e8f0", fontsize=9)

    plt.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Shortlist Threshold", 0.0, 1.0, 0.15, 0.01,
                          help="Candidates with cosine score ≥ threshold are shortlisted.")
    show_charts = st.checkbox("Show Visualizations", value=True)
    show_raw = st.checkbox("Show Raw Scores Table", value=False)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This tool uses **TF-IDF** vectorisation and **Cosine Similarity** "
        "to rank resumes against a job description — powered by NLP preprocessing "
        "(tokenisation, stopword removal, lemmatisation)."
    )

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">📄 AI Resume Screener</div>
  <div class="hero-sub">Rank candidates intelligently using NLP · TF-IDF · Cosine Similarity</div>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ────────────────────────────────────────────────────────────
col_jd, col_res = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown('<div class="section-heading">Job Description</div>', unsafe_allow_html=True)
    jd_default = (
        "We are looking for a Senior Python Developer with strong experience in machine learning\n"
        "and natural language processing. The ideal candidate should have:\n"
        "- Proficiency in Python, scikit-learn, NLTK, and spaCy\n"
        "- Experience building and deploying REST APIs (FastAPI or Flask)\n"
        "- Knowledge of TF-IDF, word embeddings, and text classification\n"
        "- Strong SQL skills (PostgreSQL or MySQL)\n"
        "- Familiarity with Docker and cloud deployment (AWS or GCP)\n"
        "- Experience with data preprocessing, feature extraction, and model evaluation\n"
        "- Good understanding of cosine similarity and information retrieval"
    )
    job_description = st.text_area("Paste the job description here", value=jd_default, height=220)

with col_res:
    st.markdown('<div class="section-heading">Candidate Resumes</div>', unsafe_allow_html=True)
    num_candidates = st.number_input("Number of Candidates", min_value=1, max_value=20, value=5, step=1)
    st.caption("Enter each candidate's name and resume text below.")

# ─── Dynamic candidate inputs ─────────────────────────────────────────────────
DEFAULTS = [
    {
        "name": "Alice Chen",
        "text": (
            "Senior Python developer with 5 years of experience in machine learning and NLP. "
            "Expert in scikit-learn, NLTK, spaCy, and transformers. Built multiple REST APIs "
            "using FastAPI and Flask. Deep knowledge of TF-IDF, cosine similarity, word embeddings, "
            "and text classification pipelines. Proficient in SQL with PostgreSQL and MySQL. "
            "Deployed models to AWS using Docker containers. Strong background in data preprocessing, "
            "feature extraction, and model evaluation using precision, recall, and F1-score. "
            "Also experienced with information retrieval systems and recommendation engines."
        ),
    },
    {
        "name": "Bob Martin",
        "text": (
            "Java backend engineer with 7 years of experience. Expert in Spring Boot, Hibernate, "
            "and Kafka. Some Python scripting for data automation. Familiar with REST API design "
            "and microservices architecture. MySQL database administration. Experience with "
            "Agile/Scrum methodologies. Built e-commerce and fintech platforms. Basic knowledge "
            "of machine learning concepts. No NLP or scikit-learn experience."
        ),
    },
    {
        "name": "Sara Kim",
        "text": (
            "Data scientist specializing in natural language processing and text analytics. "
            "Python expert with deep knowledge of scikit-learn, NLTK, spaCy, and Hugging Face "
            "transformers. Built text classification, TF-IDF based search, and information "
            "retrieval systems. Strong SQL and PostgreSQL skills. FastAPI REST APIs for ML model "
            "serving. Docker and GCP deployment experience. Published research on cosine similarity "
            "and word embeddings. Excellent data preprocessing and feature extraction skills."
        ),
    },
    {
        "name": "David Park",
        "text": (
            "Full-stack web developer with 4 years of experience in React, Node.js, and MongoDB. "
            "Some Python and Django experience for backend APIs. Familiar with basic machine "
            "learning using TensorFlow for image classification. SQL basics with MySQL. "
            "No significant NLP or text processing background. AWS deployment via Elastic Beanstalk. "
            "Strong frontend skills in JavaScript, TypeScript, and CSS."
        ),
    },
    {
        "name": "Emily Ross",
        "text": (
            "Machine learning engineer with 3 years of experience. Proficient in Python, "
            "scikit-learn, and pandas. Built NLP pipelines for sentiment analysis using NLTK. "
            "Some experience with TF-IDF and cosine similarity for document matching. "
            "Flask APIs for model deployment. PostgreSQL and basic Docker usage. GCP exposure. "
            "Data preprocessing and feature engineering experience. Currently deepening "
            "knowledge of spaCy and FastAPI."
        ),
    },
]

resumes = []
for i in range(int(num_candidates)):
    with st.expander(f"Candidate {i+1}", expanded=(i < 3)):
        c1, c2 = st.columns([1, 3])
        default_name = DEFAULTS[i]["name"] if i < len(DEFAULTS) else f"Candidate {i+1}"
        default_text = DEFAULTS[i]["text"] if i < len(DEFAULTS) else ""
        with c1:
            name = st.text_input("Name", value=default_name, key=f"name_{i}")
        with c2:
            text = st.text_area("Resume Text", value=default_text, height=130, key=f"text_{i}")
        if name.strip() and text.strip():
            resumes.append({"name": name.strip(), "text": text.strip()})

# ─── Run Button ───────────────────────────────────────────────────────────────
st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("🚀 Screen Resumes", use_container_width=True)

# ─── Results ──────────────────────────────────────────────────────────────────
if run:
    if not job_description.strip():
        st.error("Please enter a job description.")
    elif len(resumes) < 1:
        st.error("Please add at least one candidate resume.")
    else:
        with st.spinner("Analysing resumes…"):
            results_df, decision, jd_features, engine = run_screening(
                job_description, resumes, threshold
            )

        st.markdown("---")
        st.markdown("## 🏆 Ranking Results")

        # ── Summary metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Candidates", len(resumes))
        m2.metric("Shortlisted", len(decision["shortlisted"]))
        m3.metric("Rejected", len(decision["rejected"]))
        m4.metric("Top Score", f"{results_df['Cosine Score'].max():.4f}")

        st.markdown("")

        # ── Ranked cards ──
        st.markdown('<div class="section-heading">Candidate Rankings</div>', unsafe_allow_html=True)
        for _, row in results_df.iterrows():
            is_short = row["Candidate"] in decision["shortlisted"]
            tag_html = '<span class="tag-shortlist">✔ Shortlisted</span>' if is_short else '<span class="tag-reject">✘ Rejected</span>'
            kw_html = "".join(f'<span class="kw-chip">{k}</span>' for k in row["Top Keywords"].split(", ") if k)
            bar_pct = row["Match %"]
            st.markdown(f"""
            <div class="glass-card">
              <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <div>
                  <span class="rank-badge">#{int(row['Rank'])}</span>
                  <span style="color:#e2e8f0; font-weight:600; font-size:1.05rem;">{row['Candidate']}</span>
                  &nbsp;{tag_html}
                </div>
                <div style="color:#a5b4fc; font-weight:700; font-size:1.1rem;">
                  {row['Cosine Score']:.4f} &nbsp;<span style="color:#64748b; font-size:0.85rem; font-weight:400;">cosine</span>
                </div>
              </div>
              <div style="margin-top:10px;">
                <span style="color:#94a3b8; font-size:0.83rem;">Match: {bar_pct:.1f}%</span>
                <div class="match-bar-bg">
                  <div class="match-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
              </div>
              <div style="margin-top:10px;">{kw_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Decision summary ──
        st.markdown('<div class="section-heading">Screening Decision</div>', unsafe_allow_html=True)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.success("**Shortlisted**\n\n" + "\n".join(f"• {n}" for n in decision["shortlisted"]))
        with dcol2:
            if decision["rejected"]:
                st.error("**Rejected**\n\n" + "\n".join(f"• {n}" for n in decision["rejected"]))
            else:
                st.info("No candidates were rejected at the current threshold.")

        # ── Raw table ──
        if show_raw:
            st.markdown('<div class="section-heading">Raw Data Table</div>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)

        # ── Visualisations ──
        if show_charts:
            st.markdown("---")
            st.markdown("## 📊 Visualisations")
            v1, v2 = st.columns(2)
            with v1:
                st.markdown('<div class="section-heading">Match Score Bar Chart</div>', unsafe_allow_html=True)
                fig1 = make_bar_chart(results_df, threshold)
                st.pyplot(fig1, use_container_width=True)
            with v2:
                st.markdown('<div class="section-heading">TF-IDF Weight Comparison</div>', unsafe_allow_html=True)
                fig2 = make_tfidf_chart(results_df, engine, resumes)
                st.pyplot(fig2, use_container_width=True)

        # ── JD Features ──
        with st.expander("📝 Job Description NLP Analysis"):
            f1, f2, f3 = st.columns(3)
            f1.metric("Total Tokens", jd_features["total_tokens"])
            f2.metric("Unique Tokens", jd_features["unique_tokens"])
            f3.metric("Lexical Diversity", jd_features["lexical_diversity"])
            st.markdown("**Top 10 JD Keywords:**")
            kw_html = "".join(f'<span class="kw-chip">{t} ({s:.3f})</span>' for t, s in jd_features["top_terms"])
            st.markdown(kw_html, unsafe_allow_html=True)
