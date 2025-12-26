import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import implicit
import numpy as np
from scipy import sparse
from implicit.nearest_neighbours import bm25_weight, tfidf_weight, CosineRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    reciprocal_rank_at_k,
)
from data.load_movielens import load_ratings
from data.split import temporal_train_val_test_split
from data.encoding import build_id_mappings
from data.matrix import build_interaction_matrix
from models.als import ALSRecommender
import models.als as als_module

# -----------------------------
# Phase 3 Step 2 controls
# -----------------------------
EVAL_SPLIT = "val"  # "val" for development; keep "test" for final reports only
REPORT_TEST_FOR_BASELINE_V2 = True

# If True, also run the additional "signal experiments" (previous EXP 1 & 3).
# Keep this False for day-to-day work; it saves a lot of time.
RUN_SIGNAL_EXPS = False

# Baseline_v2 (frozen)
BASELINE_V2 = {
    "rating_threshold": 4.0,
    "bm25_on": False,
    "als_params": {"factors": 64, "regularization": 0.01, "iterations": 15},
}

# Eval constants
K = 10
SEED = 42
MAX_USERS = 50_000
# -----------------------------
# Phase 4A (Diversity) offline eval controls
# -----------------------------
DIVERSITY_LAMBDAS = [0.1, 0.2, 0.3]
DIVERSITY_CANDIDATES_MULT = 20  # request more ALS candidates then rerank

RUNS_PATH = PROJECT_ROOT / "runs.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "baseline_v2"

# -----------------------------
# Helper: Export baseline_v2 artifact bundle for serving
# -----------------------------
def _export_baseline_v2_artifacts(
    *,
    artifact_dir: Path,
    thr: float,
    bm25_on: bool,
    als_params: dict,
    model,
    user_item_csr,
    user_to_index: dict,
    item_to_index: dict,
    index_to_item: dict,
    train_df,
):
    """
    Export a minimal artifact bundle for serving:
      - model.pkl: pickled implicit ALS model
      - mappings.json: user_to_index, item_to_index, index_to_item
      - user_item_csr.npz: CSR matrix for filtering already-seen items
      - popularity.json: list of popular movieIds for cold-start fallback
      - metadata.json: params + shapes
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model
    with (artifact_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    # 2) Mappings (JSON-friendly)
    mappings = {
        "user_to_index": {str(k): int(v) for k, v in user_to_index.items()},
        "item_to_index": {str(k): int(v) for k, v in item_to_index.items()},
        "index_to_item": {str(k): int(v) for k, v in index_to_item.items()},
    }
    (artifact_dir / "mappings.json").write_text(
        json.dumps(mappings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 3) User-item CSR (for seen-item filtering in API)
    user_item_csr = user_item_csr.tocsr()
    sparse.save_npz(artifact_dir / "user_item_csr.npz", user_item_csr)

    # 4) Popularity fallback (from train positives)
    pop = (
        train_df.groupby("movieId")["userId"]
        .count()
        .sort_values(ascending=False)
        .head(2000)
        .index.astype(int)
        .tolist()
    )
    (artifact_dir / "popularity.json").write_text(
        json.dumps({"movieIds": pop}, indent=2) + "\n", encoding="utf-8"
    )

    # 5) Metadata
    meta = {
        "dataset": "MovieLens 25M",
        "rating_threshold": thr,
        "bm25_on": bm25_on,
        "als_params": als_params,
        "user_item_shape": [int(user_item_csr.shape[0]), int(user_item_csr.shape[1])],
        "n_users": int(user_item_csr.shape[0]),
        "n_items": int(user_item_csr.shape[1]),
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (artifact_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[ARTIFACTS] Exported baseline_v2 bundle to: {artifact_dir}")


def _load_runs(path: Path):
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    data = json.loads(txt)
    return data if isinstance(data, list) else []


def _append_run(record: dict, path: Path):
    runs = _load_runs(path)
    runs.append(record)
    path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _now():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _prepare_eval_users(df, user_to_index, *, max_users: int = MAX_USERS):
    users = df["userId"].unique()
    users = np.array([u for u in users if u in user_to_index], dtype=users.dtype)

    np.random.seed(SEED)
    if len(users) > max_users:
        users = np.random.choice(users, size=max_users, replace=False)
    return users


def _prepare_truth_sets(df, item_to_index):
    user_to_items = df.groupby("userId")["movieId"].apply(set).to_dict()
    # Filter to train universe
    return {u: {m for m in ms if m in item_to_index} for u, ms in user_to_items.items()}


def evaluate_recs(
    get_recs_fn,
    users: np.ndarray,
    user_to_items: dict,
    *,
    k: int = 10,
    debug_left: int = 0,
    pop_rank: dict | None = None,
    pop_size: int | None = None,
):
    precs, recalls, ndcgs = [], [], []
    aps, rrs = [], []
    pop_avgs = []
    skipped_empty_true = 0
    skipped_empty_recs = 0
    hit_users = 0

    for idx, user_id in enumerate(users, start=1):
        true_items = user_to_items.get(user_id, set())
        if not true_items:
            skipped_empty_true += 1
            continue

        recs = get_recs_fn(user_id)
        if not recs:
            skipped_empty_recs += 1
            continue

        if pop_rank is not None:
            tail = int(pop_size) if pop_size is not None else (max(pop_rank.values()) + 1 if pop_rank else 0)
            ranks = [int(pop_rank.get(int(x), tail)) for x in recs[:k]]
            if ranks:
                pop_avgs.append(float(np.mean(ranks)))

        p = precision_at_k(recs, true_items, k=k)
        r = recall_at_k(recs, true_items, k=k)
        nd = ndcg_at_k(recs, true_items, k=k)
        ap = average_precision_at_k(recs, true_items, k=k)
        rr = reciprocal_rank_at_k(recs, true_items, k=k)

        if any(item in true_items for item in recs[:k]):
            hit_users += 1

        if debug_left > 0:
            debug_left -= 1
            print(f"[DEBUG] user_id={user_id} true={list(true_items)[:3]} recs={recs[:10]}")

        precs.append(p)
        recalls.append(r)
        ndcgs.append(nd)
        aps.append(ap)
        rrs.append(rr)

        if idx % 1000 == 0:
            print(f"Evaluated {idx} users...")

    return {
        "precision@10": float(np.mean(precs)) if precs else 0.0,
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "map@10": float(np.mean(aps)) if aps else 0.0,
        "mrr@10": float(np.mean(rrs)) if rrs else 0.0,
        "avg_pop_rank@10": float(np.mean(pop_avgs)) if pop_avgs else 0.0,
        "n_users_eval": int(len(precs)),
        "hit_users": int(hit_users),
        "skipped_empty_true": int(skipped_empty_true),
        "skipped_empty_recs": int(skipped_empty_recs),
    }
def build_popularity(train_df, *, topn: int = 2000):
    pop = (
        train_df.groupby("movieId")["userId"]
        .count()
        .sort_values(ascending=False)
        .head(topn)
        .index.astype(int)
        .tolist()
    )
    pop_rank = {mid: rank for rank, mid in enumerate(pop)}  # rank 0 = most popular
    return pop, pop_rank


def fit_itemknn(user_item_csr, *, weight: str):
    user_item_csr = user_item_csr.tocsr()
    item_user = user_item_csr.T.tocsr()

    # KNN needs weighting to be non-degenerate at scale
    if weight == "bm25":
        item_user = bm25_weight(item_user, K1=100, B=0.8)
    elif weight == "tfidf":
        item_user = tfidf_weight(item_user)

    model = CosineRecommender(K=200)
    model.fit(item_user)
    return model


def make_itemknn_recommender(model, user_item_csr, user_to_index, index_to_item, *, k: int):
    user_item_csr = user_item_csr.tocsr()

    def _recommend(user_raw_id: int):
        if user_raw_id not in user_to_index:
            return []
        user_idx = int(user_to_index[user_raw_id])

        try:
            item_indices, _ = model.recommend(
                user_idx,
                user_item_csr,
                N=max(k * 3, k + 20),
                filter_already_liked_items=True,
            )
        except ValueError:
            user_row = user_item_csr[user_idx]
            item_indices, _ = model.recommend(
                0,
                user_row,
                N=max(k * 3, k + 20),
                recalculate_user=True,
            )
            seen = set(user_row.indices)
            item_indices = [int(i) for i in item_indices if int(i) not in seen]

        mapped = []
        for i in item_indices:
            raw = index_to_item.get(int(i))
            if raw is not None:
                mapped.append(int(raw))
        return mapped[:k]

    return _recommend


def make_als_diverse_recommender(
    *,
    implicit_model,
    user_item_csr,
    user_to_index,
    item_to_index,
    index_to_item,
    popularity: list[int],
    pop_rank: dict[int, int],
    k: int,
    lam: float,
    candidates_mult: int,
):
    user_item_csr = user_item_csr.tocsr()

    def _adj_score(raw_id: int, als_score: float) -> float:
        rank = int(pop_rank.get(raw_id, len(popularity)))
        denom = float(max(1, len(popularity)))
        penalty = 1.0 - (((rank + 1) ** 0.5) / ((denom + 1) ** 0.5))
        return float(als_score) - float(lam) * float(penalty)

    def _recommend(user_raw_id: int):
        if user_raw_id not in user_to_index:
            return popularity[:k]

        user_idx = int(user_to_index[user_raw_id])
        user_row = user_item_csr[user_idx]

        N = max(k * 5, k + 50, k * int(candidates_mult))

        try:
            item_idx, scores = implicit_model.recommend(
                0,
                user_row,
                N=N,
                recalculate_user=True,
            )
        except TypeError:
            item_idx, scores = implicit_model.recommend(
                0,
                user_row,
                N=N,
                recalculate_user=True,
                filter_already_liked_items=False,
            )

        seen_internal = set(user_row.indices)

        candidates = []
        for ii, s in zip(item_idx, scores):
            ii = int(ii)
            if ii in seen_internal:
                continue
            raw = index_to_item.get(ii)
            if raw is None:
                continue
            raw = int(raw)
            candidates.append((raw, float(s)))

        if not candidates:
            return popularity[:k]

        candidates.sort(key=lambda x: _adj_score(x[0], x[1]), reverse=True)

        recs = []
        for raw, _s in candidates:
            if raw in recs:
                continue
            recs.append(raw)
            if len(recs) >= k:
                break

        if len(recs) < k:
            for raw in popularity:
                if raw in recs:
                    continue
                ii = item_to_index.get(raw)
                if ii is None:
                    continue
                if int(ii) in seen_internal:
                    continue
                recs.append(int(raw))
                if len(recs) >= k:
                    break

        return recs[:k]

    return _recommend


def log_run(
    *,
    model_name: str,
    model_impl: str,
    model_params: dict,
    recommendation_mode: str,
    dataset: str,
    rating_threshold: float,
    bm25_on: bool,
    bm25_params: dict | None,
    k_eval: int,
    subsample_users: int,
    metrics: dict,
    notes: str,
    split: str,
    tag: str | None = None,
):
    record = {
        "run_id": f"{model_name}_movielens25m_{_now()}_thr{rating_threshold}_bm25{int(bm25_on)}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "dataset": dataset,
        "task": "Top-K recommendation",
        "split": split,
        "evaluation": {
            "k": k_eval,
            "subsample_users": subsample_users,
            "random_seed": SEED,
            "n_users_eval": metrics["n_users_eval"],
            "hit_users": metrics["hit_users"],
            "skipped_empty_true": metrics["skipped_empty_true"],
            "skipped_empty_recs": metrics["skipped_empty_recs"],
        },
        "preprocessing": {
            "implicit_feedback": True,
            "rating_threshold": rating_threshold,
            "use_ratings": False,
            "weighting": ({"method": "bm25", **bm25_params} if bm25_on else {"method": "none"}),
        },
        "model": {
            "name": model_name,
            "implementation": model_impl,
            "params": model_params,
            "recommendation_mode": recommendation_mode,
        },
        "metrics": {
            "precision@10": metrics.get("precision@10", 0.0),
            "recall@10": metrics.get("recall@10", 0.0),
            "ndcg@10": metrics.get("ndcg@10", 0.0),
            "map@10": metrics.get("map@10", 0.0),
            "mrr@10": metrics.get("mrr@10", 0.0),
            "avg_pop_rank@10": metrics.get("avg_pop_rank@10", 0.0),
        },
        "notes": notes,
    }
    if tag is not None:
        record["tag"] = tag
    _append_run(record, RUNS_PATH)


def main():
    print("implicit version:", implicit.__version__)
    print("ALS module loaded from:", als_module.__file__)

    print("Loading ratings...")
    ratings = load_ratings()

    print("Splitting train/val/test (per-user temporal)...")
    train_df_full, val_df_full, test_df_full = temporal_train_val_test_split(ratings)

    # ------------------------------------
    # Experiment selection (keep it tight)
    # ------------------------------------
    baseline_exp = {
        "rating_threshold": BASELINE_V2["rating_threshold"],
        "bm25_on": BASELINE_V2["bm25_on"],
        "bm25": None,
    }

    # Optional: older signal experiments (EXP 1 & 3) — keep for occasional comparisons only
    signal_exps = [
        {"rating_threshold": 3.5, "bm25_on": True, "bm25": {"K1": 100, "B": 0.8}},
        {"rating_threshold": 3.5, "bm25_on": False, "bm25": None},
    ]

    experiments = [baseline_exp] + (signal_exps if RUN_SIGNAL_EXPS else [])

    als_best = dict(BASELINE_V2["als_params"])

    for exp_i, exp in enumerate(experiments, start=1):
        thr = float(exp["rating_threshold"])
        bm25_on = bool(exp["bm25_on"])
        bm25_params = exp["bm25"]

        print("\n" + "=" * 72)
        is_baseline_v2 = (thr == BASELINE_V2["rating_threshold"] and bm25_on == BASELINE_V2["bm25_on"])
        exp_tag = "baseline_v2" if is_baseline_v2 else "signal"
        print(f"[EXP {exp_i}/{len(experiments)}] ({exp_tag}) threshold={thr}, bm25_on={bm25_on}")

        # Filter positives consistently across splits
        train_df = train_df_full[train_df_full["rating"] >= thr].copy()
        val_df = val_df_full[val_df_full["rating"] >= thr].copy()
        test_df = test_df_full[test_df_full["rating"] >= thr].copy()
        popularity, pop_rank = build_popularity(train_df, topn=2000)

        print("Building ID mappings from train...")
        user_to_index, index_to_user, item_to_index, index_to_item = build_id_mappings(
            train_df, user_col="userId", item_col="movieId"
        )

        print("Building user-item interaction matrix (CSR)...")
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
        ).tocsr()

        # BM25 weighting is part of the experiment signal (affects ALS training matrix)
        if bm25_on:
            user_item = bm25_weight(user_item, K1=bm25_params["K1"], B=bm25_params["B"]).tocsr()

        # Choose dev split
        dev_df = val_df if EVAL_SPLIT == "val" else test_df
        dev_users = _prepare_eval_users(dev_df, user_to_index, max_users=MAX_USERS)
        dev_truth = _prepare_truth_sets(dev_df, item_to_index)

        print(f"Evaluating on {len(dev_users)} {EVAL_SPLIT} users (seed={SEED}, max={MAX_USERS}).")

        # -------------------------
        # ALS (dev)
        # -------------------------
        print("Fitting ALS (fixed params) ...")
        als_model = ALSRecommender(
            factors=als_best["factors"],
            regularization=als_best["regularization"],
            iterations=als_best["iterations"],
            use_gpu=False,
            k=K,
        )
        als_model.fit(
            user_item_matrix=user_item,
            user_to_index=user_to_index,
            index_to_user=index_to_user,
            item_to_index=item_to_index,
            index_to_item=index_to_item,
        )

        # Export artifacts for serving (only for baseline_v2 configuration)
        if is_baseline_v2:
            _export_baseline_v2_artifacts(
                artifact_dir=ARTIFACT_DIR,
                thr=thr,
                bm25_on=bm25_on,
                als_params=als_best,
                model=als_model.model,
                user_item_csr=user_item,
                user_to_index=user_to_index,
                item_to_index=item_to_index,
                index_to_item=index_to_item,
                train_df=train_df,
            )

        print("Evaluating ALS (dev)...")
        als_metrics = evaluate_recs(
            get_recs_fn=als_model.recommend,
            users=dev_users,
            user_to_items=dev_truth,
            k=K,
            debug_left=2 if RUN_SIGNAL_EXPS else 0,
            pop_rank=pop_rank,
            pop_size=len(popularity),
        )
        print("=== ALS results (dev) ===")
        for kname, v in als_metrics.items():
            print(f"{kname}: {v}")

        log_run(
            model_name="als",
            model_impl="implicit.als.AlternatingLeastSquares",
            model_params=als_best,
            recommendation_mode="fast_then_fallback",
            dataset="MovieLens 25M",
            rating_threshold=thr,
            bm25_on=bm25_on,
            bm25_params=bm25_params,
            k_eval=K,
            subsample_users=int(len(dev_users)),
            metrics=als_metrics,
            notes=f"Dev evaluation on {EVAL_SPLIT} split (do not tune on test).",
            split=EVAL_SPLIT,
        )

        # -------------------------
        # ItemKNN (dev baseline only)
        # -------------------------
        print("Fitting ItemKNN (cosine) ...")
        knn_weight = "bm25" if bm25_on else "tfidf"
        knn_model = fit_itemknn(user_item, weight=knn_weight)
        knn_recommend = make_itemknn_recommender(
            model=knn_model,
            user_item_csr=user_item,
            user_to_index=user_to_index,
            index_to_item=index_to_item,
            k=K,
        )

        print("Evaluating ItemKNN (dev)...")
        knn_metrics = evaluate_recs(
            get_recs_fn=knn_recommend,
            users=dev_users,
            user_to_items=dev_truth,
            k=K,
            debug_left=0,
            pop_rank=pop_rank,
            pop_size=len(popularity),
        )
        # -------------------------
        # ALS + Diversity rerank (DEV) — baseline_v2 experiment only
        # -------------------------
        if is_baseline_v2 and EVAL_SPLIT == "val":
            for lam in DIVERSITY_LAMBDAS:
                print(f"Evaluating ALS + Diversity rerank (val) ... lambda={lam}, candidates_mult={DIVERSITY_CANDIDATES_MULT}")
                diverse_recommend = make_als_diverse_recommender(
                    implicit_model=als_model.model,
                    user_item_csr=user_item,
                    user_to_index=user_to_index,
                    item_to_index=item_to_index,
                    index_to_item=index_to_item,
                    popularity=popularity,
                    pop_rank=pop_rank,
                    k=K,
                    lam=float(lam),
                    candidates_mult=int(DIVERSITY_CANDIDATES_MULT),
                )

                diverse_metrics = evaluate_recs(
                    get_recs_fn=diverse_recommend,
                    users=dev_users,
                    user_to_items=dev_truth,
                    k=K,
                    debug_left=0,
                    pop_rank=pop_rank,
                    pop_size=len(popularity),
                )

                print("=== ALS+DIVERSE results (dev) ===")
                for kname, v in diverse_metrics.items():
                    print(f"{kname}: {v}")

                log_run(
                    model_name="als_diverse",
                    model_impl="implicit.als.AlternatingLeastSquares + pop-penalty rerank",
                    model_params={
                        **als_best,
                        "diversity_lambda": float(lam),
                        "candidates_mult": int(DIVERSITY_CANDIDATES_MULT),
                    },
                    recommendation_mode="als_candidates_then_rerank",
                    dataset="MovieLens 25M",
                    rating_threshold=thr,
                    bm25_on=bm25_on,
                    bm25_params=bm25_params,
                    k_eval=K,
                    subsample_users=int(len(dev_users)),
                    metrics=diverse_metrics,
                    notes="Dev evaluation on val split with popularity-penalty rerank (A/B vs ALS).",
                    split=EVAL_SPLIT,
                    tag="diversity_rerank",
                )
        print("=== ItemKNN results (dev) ===")
        for kname, v in knn_metrics.items():
            print(f"{kname}: {v}")

        log_run(
            model_name="itemknn_cosine",
            model_impl="implicit.nearest_neighbours.CosineRecommender",
            model_params={"K": 200, "fit_weight": knn_weight},
            recommendation_mode="standard_recommend_then_fallback",
            dataset="MovieLens 25M",
            rating_threshold=thr,
            bm25_on=bm25_on,
            bm25_params=bm25_params,
            k_eval=K,
            subsample_users=int(len(dev_users)),
            metrics=knn_metrics,
            notes=f"Dev baseline on {EVAL_SPLIT} split (for reference only).",
            split=EVAL_SPLIT,
        )

        # -------------------------
        # TEST report only for baseline_v2 (ALS only)
        # -------------------------
        if REPORT_TEST_FOR_BASELINE_V2 and is_baseline_v2:
            test_users = _prepare_eval_users(test_df, user_to_index, max_users=MAX_USERS)
            test_truth = _prepare_truth_sets(test_df, item_to_index)
            print(f"[BASELINE_V2 TEST] Evaluating ALS on {len(test_users)} test users...")

            test_metrics = evaluate_recs(
                get_recs_fn=als_model.recommend,
                users=test_users,
                user_to_items=test_truth,
                k=K,
                debug_left=0,
                pop_rank=pop_rank,
                pop_size=len(popularity),
            )
            print("=== BASELINE_V2 TEST (ALS) ===")
            for kname, v in test_metrics.items():
                print(f"{kname}: {v}")

            log_run(
                model_name="als",
                model_impl="implicit.als.AlternatingLeastSquares",
                model_params=als_best,
                recommendation_mode="fast_then_fallback",
                dataset="MovieLens 25M",
                rating_threshold=thr,
                bm25_on=bm25_on,
                bm25_params=bm25_params,
                k_eval=K,
                subsample_users=int(len(test_users)),
                metrics=test_metrics,
                notes="BASELINE_V2 final test report (do not tune on this).",
                split="test",
                tag="baseline_v2_report",
            )

    print("\n" + "=" * 72)
    print(f"Done. Appended results to {RUNS_PATH}")


if __name__ == "__main__":
    main()
