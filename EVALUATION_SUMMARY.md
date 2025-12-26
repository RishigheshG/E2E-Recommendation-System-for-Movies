# Offline Evaluation Summary — MovieLens 25M

This document summarizes the offline evaluation of multiple recommendation baselines
on the MovieLens 25M dataset using implicit feedback.

The goal is to compare model behavior, not to optimize leaderboard scores.

---

## Dataset & Setup

- Dataset: MovieLens 25M
- Feedback type: Implicit (rating thresholding)
- Thresholds tested: 3.5, 4.0
- Splits: Train / Validation / Test (strict separation)
- Evaluation users: Subsampled up to 50,000 users
- Recommendation task: Top-10 ranking

---

## Models Evaluated

1. **Popularity Baseline**
   - Most-interacted movies globally
   - Used as cold-start fallback

2. **ItemKNN (Cosine)**
   - Item-item similarity using implicit interactions
   - Included as a reference baseline

3. **ALS (Matrix Factorization)**
   - Alternating Least Squares using `implicit`
   - Tuned on validation split
   - Main backbone model

4. **ALS + Diversity Re-ranking**
   - Greedy re-ranking applied on ALS candidates
   - Trade-off between relevance and novelty

---

## Evaluation Metrics

- Precision@10
- Recall@10
- NDCG@10
- MAP@10
- MRR@10

Metrics are computed per-user and averaged.

---

## Key Results (Validation Split)

| Model | Precision@10 | Recall@10 | NDCG@10 | Notes |
|-----|-------------|-----------|---------|------|
| Popularity | Low | Low | Low | Simple baseline |
| ItemKNN | ~0 | ~0 | ~0 | Sparse, long-tail failure |
| ALS | Best | Best | Best | Strong latent signal |
| ALS + Diversity | Slightly ↓ | Slightly ↓ | Slightly ↓ | Improved novelty |

*(Exact values are logged in `runs.json`)*

---

## Interpretation

### Why ItemKNN performed poorly
- MovieLens 25M has a long-tail item distribution
- Item-item similarity is extremely sparse
- Many users have weak co-interaction signals
- Result: almost no overlap with held-out positives

### Why ALS outperformed baselines
- Captures latent user–item structure
- Shares information across sparse interactions
- Scales well to large implicit datasets

### Effect of Diversity Re-ranking
- Re-ranking reduces pure relevance metrics
- Increases catalog coverage and novelty
- Demonstrates a real-world trade-off between accuracy and discovery

---

## Takeaways

- Strong baselines are essential before complex models
- Offline metrics must be interpreted, not blindly optimized
- Matrix factorization is a solid default for implicit feedback
- Diversity mechanisms should be evaluated deliberately

---

## Notes

This evaluation is strictly offline.
Online performance would require user interaction data and A/B testing.