# VERITAS.AI — Continuous Learning & Dataset Management Guide

## 1. Overview
VERITAS.AI includes a modular continuous-learning pipeline that discovers new articles, validates labels safely without trusting unverified search results or ML predictions as ground truth, queues unverified articles for human review, and retrains the Logistic Regression + TF-IDF model under strict holdout evaluation gates.

## 2. Configuration (Environment Variables)
- `VERITAS_ADMIN_KEY`: Administrative access key required for `/admin/dataset` and `/api/admin/*` endpoints (default dev value: `veritas-admin-2026`).
- `VERITAS_SECRET_KEY`: Flask session secret key.
- `VERITAS_COLLECTION_INTERVAL_MIN`: Periodic collection interval in minutes for APScheduler (default: `60`).
- `VERITAS_MIN_APPROVED_RETRAIN`: Minimum number of newly `APPROVED` articles required before retraining runs (default: `10`).
- `VERITAS_DISABLE_SCHEDULER`: Set to `1` to disable the background scheduler (recommended during automated tests).

## 3. Dependencies
Installed in `.venv` and listed in `requirements.txt`:
- `APScheduler==3.11.3` (periodic collection scheduling)
- `requests`, `beautifulsoup4`, `trafilatura` (article retrieval & ClaimReview extraction)
- `scikit-learn`, `nltk`, `pandas`, `joblib` (existing TF-IDF + PorterStemmer + Logistic Regression pipeline)

## 4. Dataset Storage & Anti-Leakage Architecture
- **Original Dataset (Read-Only)**: `dataset/Fake.csv` and `dataset/True.csv` are never modified or overwritten.
- **Fixed Holdout Test Set**: `data_pipeline/datasets/holdout_test.csv` is created once via a stratified 20% split (`random_state=42`) and remains untouched for all model evaluations.
- **Base Training Split**: `data_pipeline/datasets/train_base.csv` (the remaining 80% of original data).
- **Candidate & Version Store**: `data_pipeline/pipeline.db` (SQLite) tracks:
  - `candidates`: `PENDING`, `APPROVED`, `REJECTED`, `NEEDS_REVIEW`
  - `dataset_versions`: versioned training snapshots (`data_pipeline/datasets/training_v*.csv`)
  - `model_versions`: evaluation metrics (`accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`) and active deployment state.

## 5. Controlled Retraining & Rollback
- Retraining runs only when `untrained_approved >= VERITAS_MIN_APPROVED_RETRAIN`.
- Only `APPROVED` articles (deduplicated and verified absent from `holdout_test.csv`) are merged with `train_base.csv`.
- `TfidfVectorizer(max_features=5000)` and `LogisticRegression(max_iter=1000)` are fitted together and saved to `models/versions/<version>/`.
- **Deployment Gate**: A candidate model is promoted to `models/fake_news_model.pkl` & `models/tfidf_vectorizer.pkl` only if its holdout F1 and accuracy are greater than or equal to the active production model.
- **Recovery / Rollback**: Use the **Rollback** button in `/admin/dataset` (or `POST /api/admin/rollback` with `{"version": "v1.0"}`) to restore any previous `(model, vectorizer)` pair atomically.
