from math import log2
from typing import Sequence, Set

def precision_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0

    if not recommended:
        return 0.0

    recommended_k = recommended[:k]
    if not relevant:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / float(k)


def recall_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    if not relevant:
        return 0.0

    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    dcg = 0.0

    for rank, item in enumerate(recommended_k, start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    # Ideal DCG: all relevant items at top (but limited by k)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
