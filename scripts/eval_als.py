import sys
from pathlib import Path

import numpy as np
from implicit.nearest_neighbours import bm25_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.load_movielens import load_ratings
from data.split import temporal_train_val_test_split
from data.encoding import build_id_mappings
from data.matrix import build_interaction_matrix
from models.als import ALSRecommender
import models.als as als_module
from eval.metrics import precision_at_k, recall_at_k, ndcg_at_k

K = 10


def main():
    print("Loading ratings...")
    ratings = load_ratings()  # full load; if it's too slow, use nrows=...

    print("Splitting train/val/test...")
    train_df, val_df, test_df = temporal_train_val_test_split(ratings)

    # For implicit ALS, treat only strong positive ratings as interactions
    train_df = train_df[train_df["rating"] >= 4.0].copy()
    test_df = test_df[test_df["rating"] >= 4.0].copy()

    print("Building ID mappings from train...")
    user_to_index, index_to_user, item_to_index, index_to_item = build_id_mappings(
        train_df, user_col="userId", item_col="movieId"
    )

    print("Building user-item interaction matrix...")
    user_item = build_interaction_matrix(
        train_df,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        user_col="userId",
        item_col="movieId",
        rating_col="rating",
        use_ratings=False,
        alpha=40.0,
        orientation="user_item",
    )

    # Apply BM25 weighting on the user×item matrix
    user_item = bm25_weight(user_item, K1=100, B=0.8)

    print("Fitting ALS model (CPU) ...")
    print("ALS module loaded from:", als_module.__file__)
    als_model = ALSRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        use_gpu=False,  # M1 -> CPU
        k=K,
    )
    als_model.fit(
        user_item_matrix=user_item,
        user_to_index=user_to_index,
        index_to_user=index_to_user,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
    )

    print("Evaluating ALS on test users...")

    # Diagnostics: confirm shapes/mappings are consistent before recommending
    print("n_users_train_mapping:", len(user_to_index))
    print("n_items_train_mapping:", len(item_to_index))
    print("user_item.shape:", user_item.shape)
    print("als user_item shape:", als_model.user_item_csr.shape)
    print("model.user_factors:", als_model.model.user_factors.shape)
    print("model.item_factors:", als_model.model.item_factors.shape)

    test_users = test_df["userId"].unique()
    print(
        "max internal user_idx in sample:",
        max(user_to_index[u] for u in test_users if u in user_to_index),
    )

    # Only evaluate users seen during training (ALS can't score cold users)
    test_users = np.array([u for u in test_users if u in user_to_index], dtype=test_users.dtype)

    # Subsample for dev on M1 so you don't sit forever
    np.random.seed(42)
    max_users = 10_000
    if len(test_users) > max_users:
        test_users = np.random.choice(test_users, size=max_users, replace=False)
        print(f"Subsampled to {len(test_users)} test users.")

    # Build user -> true set of items
    user_to_items = (
        test_df.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    # Filter true items to those seen in training (otherwise ALS can never hit them)
    user_to_items = {u: {m for m in ms if m in item_to_index} for u, ms in user_to_items.items()}

    precs, recalls, ndcgs = [], [], []
    skipped_empty_true = 0
    skipped_empty_recs = 0
    hit_users = 0
    debug_left = 5

    for idx, user_id in enumerate(test_users, start=1):
        true_items = user_to_items.get(user_id, set())
        if not true_items:
            skipped_empty_true += 1
            continue

        recs = als_model.recommend(user_id)
        if not recs:
            skipped_empty_recs += 1
            continue

        p = precision_at_k(recs, true_items, k=K)
        r = recall_at_k(recs, true_items, k=K)
        nd = ndcg_at_k(recs, true_items, k=K)

        if any(item in true_items for item in recs[:K]):
            hit_users += 1

        if debug_left > 0:
            debug_left -= 1
            print(f"[DEBUG] user_id={user_id} true={list(true_items)[:3]} recs={recs[:10]}")

        precs.append(p)
        recalls.append(r)
        ndcgs.append(nd)

        if idx % 1000 == 0:
            print(f"Evaluated {idx} users...")

    print("=== ALS results ===")
    print("precision@10:", float(np.mean(precs)) if precs else 0.0)
    print("recall@10:", float(np.mean(recalls)) if recalls else 0.0)
    print("ndcg@10:", float(np.mean(ndcgs)) if ndcgs else 0.0)
    print("n_users_eval:", len(precs))
    print("hit_users:", hit_users)
    print("skipped_empty_true:", skipped_empty_true)
    print("skipped_empty_recs:", skipped_empty_recs)


if __name__ == "__main__":
    main()