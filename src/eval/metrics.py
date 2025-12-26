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


def average_precision_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """
    Average Precision at K (AP@K) for a single user.

    AP@K = (1 / min(|relevant|, k)) * sum_{i=1..k} [rel_i * precision@i]

    Where rel_i = 1 if the i-th recommended item is relevant, else 0.
    If there are no relevant items, returns 0.0.

    Note: MAP@K is the mean of AP@K across users.
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    if not recommended:
        return 0.0

    recommended_k = recommended[:k]

    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended_k, start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / float(i)

    denom = float(min(len(relevant), k))
    return sum_precisions / denom if denom > 0 else 0.0


def reciprocal_rank_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """
    Reciprocal Rank at K (RR@K) for a single user.

    RR@K = 1 / rank of the first relevant item within top-k, else 0.
    Note: MRR@K is the mean of RR@K across users.
    """
    if k <= 0:
        return 0.0
    if not relevant:
        return 0.0
    if not recommended:
        return 0.0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / float(rank)
    return 0.0
