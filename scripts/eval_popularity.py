import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.load_movielens import load_ratings
from data.split import temporal_train_val_test_split
from models.popularity import PopularityRecommender
from eval.metrics import precision_at_k, recall_at_k, ndcg_at_k


K = 10

def main():
    print("Loading ratings...")
    ratings = load_ratings(nrows = 5_000_000)  # full data load (if this is too slow, you can sample users for eval)

    print("Splitting train/val/test...")
    train_df, val_df, test_df = temporal_train_val_test_split(ratings)

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    print("Fitting PopularityRecommender on train...")
    pop_model = PopularityRecommender(top_k=K)
    pop_model.fit(train_df)

    print("Evaluating on test users...")

    test_users = test_df["userId"].unique()

    import numpy as np
    np.random.seed(42)

    max_users = 10_000  # or even 5_000 while developing
    if len(test_users) > max_users:
        test_users = np.random.choice(test_users, size=max_users, replace=False)
        print(f"Subsampled to {len(test_users)} test users.")

    precs = []
    recalls = []
    ndcgs = []

    # Create mapping from user -> set of true test items
    user_to_items = (
        test_df.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    for idx, user_id in enumerate(test_users, start = 1):
        true_items = user_to_items[user_id]
        recs = pop_model.recommend(user_id)

        if not recs:
            continue

        p = precision_at_k(recs, true_items, k=K)
        r = recall_at_k(recs, true_items, k=K)
        nd = ndcg_at_k(recs, true_items, k=K)

        precs.append(p)
        recalls.append(r)
        ndcgs.append(nd)

        if idx % 1000 == 0:
            print(f"Evaluated {idx} users...")

    results = {
        "precision@10": float(np.mean(precs)) if precs else 0.0,
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_eval": int(len(precs)),
    }

    print("=== Popularity baseline results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
