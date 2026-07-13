# RetainAI — Employee Attrition Prediction & Retention Engine

An end-to-end machine learning system that predicts employee attrition risk and generates
transparent, policy-backed retention recommendations — combining a deterministic decision
engine with a retrieval-augmented Q&A assistant for company policy questions.

**Live Demo:** https://sachitha001-retainai.hf.space

---

## Why This Project Exists

Most "AI HR tools" let an LLM generate advice directly, which risks hallucinated or
inconsistent recommendations for decisions that affect real people's jobs. RetainAI is built
around a deliberate separation:

- **Consequential decisions** (who is high-risk, what retention actions to recommend) are handled
  by a deterministic, rule-based policy engine — fully auditable, zero hallucination risk.
- **Open-ended questions** ("what's our parental leave policy?") are handled separately by a
  RAG pipeline (FAISS + Groq), clearly isolated from the prediction/decision path.

This mirrors how safety-conscious production systems in regulated domains (HR, finance,
healthcare) are architected — generative AI for information, deterministic logic for decisions.

---

## Architecture
Data Ingestion → Schema Validation → Transformation (ColumnTransformer: RobustScaler + OneHotEncoder)
→ SMOTE-Tomek (train only) → XGBoost Training (MLflow tracked)
→ Gradio App (Hugging Face Spaces)
├── Deterministic Policy Engine  (JSON rulebase → retention memo)
└── Policy Q&A (RAG)             (FAISS retrieval + Groq LLM, isolated tab)
Docker image built + pushed via GitHub Actions CI/CD → Docker Hub
Dataset versioned via DVC

---

## Tech Stack

- **ML:** Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE-Tomek)
- **Experiment tracking:** MLflow
- **Data versioning:** DVC
- **RAG:** LangChain, FAISS, HuggingFace sentence-transformers (`all-MiniLM-L6-v2`), Groq API (`llama-3.1-8b-instant`)
- **App/UI:** Gradio (primary, deployed), Streamlit (local monitoring dashboard)
- **CI/CD & packaging:** Docker, GitHub Actions

---

## Features

- Modular pipeline: ingestion → YAML-schema validation → transformation → training
- Class-imbalance handling via SMOTE-Tomek applied only to training data (no leakage)
- XGBoost classifier: **88% test accuracy, 0.78 recall** on the minority (attrition) class after resampling
- MLflow experiment tracking across model versions
- Deterministic, JSON-rulebase retention policy engine — no LLM involved in decision output
- RAG-powered Policy Q&A tab, fully separated from prediction logic, with retrieved context shown for transparency
- Local Streamlit dashboard for monitoring model behavior (drift, accuracy trend, latency)
- Dockerized build with automated CI/CD via GitHub Actions
- 30+ input fields organized into logical sections with one-click sample presets (Low/High/Borderline/Random risk)

---

## Quick Start

```bash
git clone https://github.com/sachithags/RetainAI.git
cd RetainAI
pip install -r requirements.txt
python main.py            # runs the training pipeline
python app.py              # launches the Gradio app
# open http://127.0.0.1:7860
```

## Docker

```bash
docker pull sachithags/retainai:latest
docker run -p 7860:7860 sachithags/retainai:latest
```

---

## Key Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 88% |
| Recall (minority class, post SMOTE-Tomek) | 0.78 |
| Baseline majority-class accuracy | ~84% (dataset attrition rate ~16%) |

*(Note: accuracy alone is not the primary metric given class imbalance — recall on the
minority class is prioritized since missing a true attrition case is costlier than a false alarm.)*

---

## Known Limitations

- Monitoring dashboard currently uses a held-out data slice as a proxy for "production" traffic,
  not live production data — it demonstrates the drift-detection *method*, not live monitoring.
- No authentication layer; the public demo is read-only and does not persist or expose real employee data.
- Trained on a single public HR dataset; not validated against a live organization's data distribution.
- No automated retraining trigger yet — model updates are manual.
