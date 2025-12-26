# End-to-End Movie Recommendation System (MovieLens 25M)

An end-to-end **recommendation system project** built to demonstrate **real-world ML engineering**, not just model training.

This project covers:
- Offline experimentation (baselines → ALS → diversity re-ranking)
- Proper evaluation on validation/test splits
- Model artifact export
- FastAPI inference service with caching and fallbacks
- Lightweight mobile demo (Expo / React Native) to showcase results

> ⚠️ **Important**: This project is for learning, experimentation, and portfolio demonstration.  
> It is **not a production consumer app**.

---

## What This Project Demonstrates

- How to build and evaluate recommender systems beyond notebooks
- Why baselines (Popularity, ItemKNN) matter before complex models
- Proper offline evaluation with strict train/val/test separation
- Trade-offs between relevance and diversity in recommendations
- How to serve ML models via an API with caching and cold-start fallbacks
- How to expose ML systems through a minimal mobile demo

This project focuses on **ML system design**, not frontend or product UX.

---

## Project Overview

**Goal:**  
Build a scalable movie recommendation system using implicit feedback, evaluate multiple baselines, and expose the final model via an API that can be consumed by a demo app.

**Dataset:**  
- MovieLens 25M (GroupLens)
- Dataset is **NOT committed** to GitHub (ignored via `.gitignore`)
- Expected locally under `data/raw/ml-25m/`

---

## Repository Structure

```
E2E-Recommendation-System-for-Movies/
│
├── src/
│   ├── api/        # FastAPI service
│   ├── data/       # Data loading & preprocessing logic
│   ├── eval/       # Evaluation metrics
│   └── models/     # ALS, ItemKNN implementations
│
├── scripts/        # Offline experiment runners (eval_als.py, etc.)
├── notebooks/      # Exploration / debugging notebooks
├── demo-app/       # Expo (React Native) demo application
│
├── data/           # Local-only datasets (ignored)
│   └── raw/ml-25m/
│
├── artifacts/      # Exported model bundles (ignored)
├── assets/
│   └── screenshots/
│
├── runs.json       # Logged experiment results
├── baseline-2.txt  # Baseline notes
├── README.md
├── .gitignore
```

---

## System Architecture

```
MovieLens 25M (local)
        ↓
Data preprocessing (implicit feedback)
        ↓
Offline models
  - Popularity baseline
  - ItemKNN (Cosine)
  - ALS (Matrix Factorization)
        ↓
Evaluation (Precision@K, Recall@K, NDCG, MAP, MRR)
        ↓
Artifact export (baseline_v2)
        ↓
FastAPI inference service
        ↓
Expo (React Native) demo app
```

---

## Models Implemented

### 1. Popularity Baseline
- Most-watched movies
- Used as cold-start fallback

### 2. ItemKNN (Cosine)
- Item–item similarity using implicit interactions
- Included mainly as a reference baseline

### 3. ALS (Alternating Least Squares)
- Matrix factorization using `implicit`
- Tuned on validation split
- Final backbone model

### 4. ALS + Diversity Re-ranking
- Greedy re-ranking to increase catalog diversity
- Adjustable diversity strength (`lambda`)
- Demonstrates trade-off between relevance and novelty

---

## Evaluation Metrics

All models are evaluated using:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- MRR@K

Validation and test splits are **strictly separated**.

---

## FastAPI Service

The FastAPI backend:
- Loads exported ALS artifacts
- Supports:
  - ALS recommendations
  - Diverse recommendations
  - Cold-start fallback
- Includes:
  - In-memory caching
  - Latency logging
  - `/health`, `/stats`, `/recommend`, `/compare` endpoints

### Run API locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:

```
http://<YOUR_LOCAL_IP>:8000
```

---

## Demo App (Expo / React Native)

A minimal demo app to **visualize recommendations**, not a full product.

Features:
- Enter a `userId`
- View:
  - ALS recommendations (accuracy-focused)
  - Diverse recommendations (discovery-focused)
- Displays movie titles & genres
- Uses FastAPI backend

> This app is intentionally minimal.  
> The goal is **model explainability and system demonstration**, not UX polish.

---

## What This Project Is (and Is Not)

### ✅ This project is:
- A serious ML engineering portfolio project
- End-to-end (data → model → API → client)
- Focused on recommender system fundamentals

### ❌ This project is NOT:
- A consumer-ready movie app
- A personalized onboarding flow
- A frontend-heavy product

---

## Key Takeaways

- Recommendation quality comes from **data + evaluation**, not UI
- Baselines matter
- Offline metrics guide system design
- Serving, caching, and fallbacks are part of ML engineering
- A simple demo is enough to showcase complex systems

---

## Next Extensions (Optional)
- Candidate generation + re-ranking
- Hybrid models (ALS + popularity)
- Online evaluation simulation
- Feature-based re-ranking

---

## Tech Stack

- Python
- NumPy / SciPy
- implicit
- FastAPI
- Uvicorn
- Expo / React Native

---

## License
Academic / educational use.