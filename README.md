# RetainAI — Employee Attrition Prediction & Retention Engine

An end‑to‑end machine learning system that predicts which employees are
likely to leave and automatically generates personalised retention plans
using a RAG pipeline grounded in company policy documents.

## Architecture

Raw HR Data (CSV) → MongoDB → ETL Pipeline → Feature Engineering →
XGBoost Model → MLflow Tracking → FastAPI Endpoint → Streamlit Dashboard

CI/CD via GitHub Actions → Docker → AWS ECR → AWS EC2

The retention plan generator uses a LangChain RAG pipeline that retrieves
relevant policy sections from a FAISS vector store and generates a
human‑readable memo with an LLM (Ollama / Gemini).

## Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit‑learn, XGBoost
- **MLOps:** MLflow, DVC, Docker, GitHub Actions
- **Cloud:** AWS EC2, AWS ECR, MongoDB Atlas
- **GenAI:** LangChain, HuggingFace Embeddings, FAISS, Ollama
- **Web:** FastAPI, Streamlit

## Project Structure

project-root/
│
├── artifacts/                    # Generated models, data, preprocessor
│
├── retainAI/
│   ├── components/               # Data ingestion, validation, transformation, training
│   ├── entity/                   # Config and artifact entities
│   ├── pipeline/                 # Training and prediction pipelines
│   └── utils/                    # Helper functions
│
├── config/
│   └── schema.yaml               # Data validation schema
│
├── constants/                    # Paths, column names
│
├── app.py                        # FastAPI app
├── streamlit_app.py              # Dashboard
├── Dockerfile
├── .github/workflows/            # CI/CD
└── README.md

## Features

- Modular ETL pipeline with data validation against a YAML schema
- SMOTE‑Tomek resampling to handle class imbalance
- XGBoost classifier achieving 91% accuracy, 0.88 ROC‑AUC
- MLflow experiment tracking with model registry
- Dockerised deployment with GitHub Actions CI/CD
- RAG‑powered retention plan generator using policy PDFs

## Setup

1. Clone the repo and create a `.env` file with MongoDB URL and AWS credentials.
2. Install dependencies: `pip install -r requirements.txt`
3. Place the HR dataset (CSV) in `RetainAI_data/`.
4. Run the pipeline: `python main.py`
5. Start the API: `uvicorn app:app --reload`
6. Launch the dashboard: `streamlit run streamlit_app.py`

## Key Metrics

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 91%       |
| ROC‑AUC       | 0.88      |
| Minority Recall (after SMOTE) | 0.78 |
| Avg API Latency | <200 ms |

## Responsible AI Note

The system is designed to support HR decision‑making, not replace it.
All AI‑generated retention plans are marked as drafts and must be reviewed
by a human before any action is taken. The model uses only business‑relevant
features and excludes protected attributes (race, religion, etc.).
