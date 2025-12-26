# Offline Evaluation Summary

## Dataset
MovieLens 25M (implicit feedback, threshold-based)

## Models Compared
- Popularity baseline
- ItemKNN (cosine)
- ALS
- ALS + Diversity Re-ranking

## Metrics
Precision@10, Recall@10, NDCG@10, MAP@10, MRR@10

## Key Results (Validation)

| Model           | Precision@10         | Recall@10   | NDCG@10   |
|-----------------|----------------------|-------------|-----------|
| Popularity      | low                  | low         | low       |
| ItemKNN         | ~0                   | ~0          | ~0        |
| ALS             | best                 | best        | best      |
| ALS + Diversity | slightly ↓ relevance | ↑ diversity | trade-off |

## Interpretation

- ItemKNN performs poorly due to sparse long-tail interactions
- ALS captures latent user-item structure effectively
- Diversity re-ranking slightly reduces relevance metrics
- Diversity improves catalog coverage and recommendation novelty