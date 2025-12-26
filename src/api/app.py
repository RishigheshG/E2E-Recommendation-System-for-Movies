from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Ensure `src/` is importable when running via uvicorn
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from data.load_movielens import load_movies

def _pick_artifact_dir(project_root: Path) -> Path:
    # Explicit override wins
    env = os.getenv("ARTIFACT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()

    base = (project_root / "artifacts").resolve()

    # Prefer versioned snapshots like baseline_v2_001, baseline_v2_002, ...
    if base.exists():
        candidates = sorted(
            [p for p in base.iterdir() if p.is_dir() and p.name.startswith("baseline_v2_")],
            key=lambda p: p.name,
        )
        if candidates:
            return candidates[-1]

    # Fallback to the dev folder
    return (base / "baseline_v2").resolve()


ART_DIR = _pick_artifact_dir(PROJECT_ROOT)

app = FastAPI(title="Movie Recommender API", version="0.1.0")

# Globals loaded at startup
MODEL = None
USER_TO_INDEX = None
ITEM_TO_INDEX = None
INDEX_TO_ITEM = None
USER_ITEM = None
POPULAR = None
# Movie metadata for demo lookups: movieId -> {"title": str, "genres": str}
MOVIE_META = None

# Popularity rank map for diversity penalty (raw movieId -> rank)
POP_RANK = None

# Diversity controls (popularity-penalty reranking)
DIVERSITY_ON = os.getenv("DIVERSITY_ON", "1").strip().lower() not in {"0", "false", "no"}
DIVERSITY_LAMBDA = float(os.getenv("DIVERSITY_LAMBDA", "0.1"))  # default validated on val
DIVERSITY_CANDIDATES_MULT = int(os.getenv("DIVERSITY_CANDIDATES_MULT", "20"))  # more candidates

# Simple in-memory cache (process-local). Good enough for demo.
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))

# key -> (expires_at_epoch, movieIds)
REC_CACHE: "OrderedDict[Tuple[int, int, str], tuple[float, List[int]]]" = OrderedDict()
CACHE_HITS = 0
CACHE_MISSES = 0

# Latency stats (ms)
LAT_SAMPLES = 0
LAT_TOTAL_MS = 0.0
LAT_MAX_MS = 0.0


class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10
    mode: Optional[str] = None  # "als" or "diverse"


def _load_artifacts():
    global MODEL, USER_TO_INDEX, ITEM_TO_INDEX, INDEX_TO_ITEM, USER_ITEM, POPULAR

    model_path = ART_DIR / "model.pkl"
    mappings_path = ART_DIR / "mappings.json"
    user_item_path = ART_DIR / "user_item_csr.npz"
    pop_path = ART_DIR / "popularity.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run: python scripts/eval_als.py")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Missing {mappings_path}.")
    if not user_item_path.exists():
        raise FileNotFoundError(f"Missing {user_item_path}.")
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing {pop_path}.")

    with model_path.open("rb") as f:
        MODEL = pickle.load(f)

    mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
    USER_TO_INDEX = {int(k): int(v) for k, v in mappings["user_to_index"].items()}
    ITEM_TO_INDEX = {int(k): int(v) for k, v in mappings["item_to_index"].items()}
    INDEX_TO_ITEM = {int(k): int(v) for k, v in mappings["index_to_item"].items()}

    USER_ITEM = sparse.load_npz(user_item_path).tocsr()

    pop = json.loads(pop_path.read_text(encoding="utf-8"))
    POPULAR = [int(x) for x in pop.get("movieIds", [])]
    # rank 0 = most popular (largest penalty)
    global POP_RANK
    POP_RANK = {mid: rank for rank, mid in enumerate(POPULAR)}

    # Load movie metadata (best-effort). Enables /movies endpoint + optional include_meta in compare.
    global MOVIE_META
    try:
        movies_df = load_movies(usecols=["movieId", "title", "genres"])
        MOVIE_META = {
            int(mid): {"title": str(title), "genres": str(genres)}
            for mid, title, genres in movies_df[["movieId", "title", "genres"]].itertuples(index=False, name=None)
        }
    except Exception as e:
        MOVIE_META = None
        print(f"[WARN] Could not load movies metadata: {e}")


def _cache_get(key: Tuple[int, int, str]):
    global CACHE_HITS, CACHE_MISSES
    now = time.time()

    if key in REC_CACHE:
        expires_at, value = REC_CACHE[key]
        if expires_at > now:
            REC_CACHE.move_to_end(key)
            CACHE_HITS += 1
            return value
        REC_CACHE.pop(key, None)

    CACHE_MISSES += 1
    return None


def _cache_set(key: Tuple[int, int, str], value: List[int]):
    expires_at = time.time() + float(CACHE_TTL_SECONDS)
    REC_CACHE[key] = (expires_at, value)
    REC_CACHE.move_to_end(key)

    while len(REC_CACHE) > CACHE_MAX_ENTRIES:
        REC_CACHE.popitem(last=False)


def _record_latency(ms: float):
    global LAT_SAMPLES, LAT_TOTAL_MS, LAT_MAX_MS
    LAT_SAMPLES += 1
    LAT_TOTAL_MS += ms
    if ms > LAT_MAX_MS:
        LAT_MAX_MS = ms


def _normalize_mode(mode: Optional[str]) -> str:
    if mode is None or str(mode).strip() == "":
        # Default: if diversity is globally enabled, use diverse; else als.
        return "diverse" if DIVERSITY_ON else "als"
    m = str(mode).strip().lower()
    if m not in {"als", "diverse"}:
        raise ValueError("mode must be 'als' or 'diverse'")
    # If diversity is globally disabled, force als
    if not DIVERSITY_ON and m == "diverse":
        return "als"
    return m


def _recommend(user_id: int, k: int, mode: str) -> List[int]:
    # Always return k, fallback to popularity fill.
    if k <= 0:
        return []

    cache_key = (int(user_id), int(k), mode)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Cold-start: stable popularity fallback
    if user_id not in USER_TO_INDEX:
        out = POPULAR[:k]
        _cache_set(cache_key, out)
        return out

    uidx = USER_TO_INDEX[user_id]

    # implicit.recommend() in this environment expects user_items to have 1 row
    # for each provided user id. We'll pass a single CSR row and recalculate.
    user_row = USER_ITEM[uidx]

    try:
        item_idx, scores = MODEL.recommend(
            0,
            user_row,
            N=max(k * 5, k + 50, k * DIVERSITY_CANDIDATES_MULT),
            recalculate_user=True,
        )
    except TypeError:
        # Older implicit versions may require explicit filter flag even with recalc
        item_idx, scores = MODEL.recommend(
            0,
            user_row,
            N=max(k * 5, k + 50, k * DIVERSITY_CANDIDATES_MULT),
            recalculate_user=True,
            filter_already_liked_items=False,
        )

    seen_internal = set(user_row.indices)

    # Filter already-seen internal item indices (keep scores aligned)
    filtered = []
    for i, s in zip(item_idx, scores):
        ii = int(i)
        if ii in seen_internal:
            continue
        filtered.append((ii, float(s)))

    # Nothing left -> fallback fill
    if not filtered:
        out = POPULAR[:k]
        _cache_set(cache_key, out)
        return out

    # Map internal -> raw and apply optional diversity reranking
    candidates = []
    for ii, s in filtered:
        raw = INDEX_TO_ITEM.get(int(ii))
        if raw is None:
            continue
        raw = int(raw)
        candidates.append((raw, s))

    if not candidates:
        out = POPULAR[:k]
        _cache_set(cache_key, out)
        return out

    if mode == "diverse" and DIVERSITY_ON and POP_RANK is not None and DIVERSITY_LAMBDA > 0:
        def adj_score(raw_id: int, als_score: float) -> float:
            # Normalize popularity to a [0,1] penalty using log scaling:
            # rank=0 -> penalty ~1.0, tail -> penalty ~0.0
            rank = POP_RANK.get(raw_id, len(POPULAR))
            denom = float(max(1, len(POPULAR)))
            penalty = 1.0 - ( ( (rank + 1) if rank >= 0 else 1 ) ** 0.5 ) / ( (denom + 1) ** 0.5 )
            return als_score - (DIVERSITY_LAMBDA * penalty)

        candidates.sort(key=lambda x: adj_score(x[0], x[1]), reverse=True)
    else:
        # Default: ALS score
        candidates.sort(key=lambda x: x[1], reverse=True)

    recs = []
    for raw, _s in candidates:
        if raw in recs:
            continue
        recs.append(raw)
        if len(recs) >= k:
            break

    if len(recs) < k:
        # Fill from popularity, avoiding seen + duplicates
        for raw in POPULAR:
            ii = ITEM_TO_INDEX.get(raw)
            if ii is None:
                continue
            if ii in seen_internal:
                continue
            if raw in recs:
                continue
            recs.append(raw)
            if len(recs) >= k:
                break

    out = recs[:k]
    _cache_set(cache_key, out)
    return out



# --- User history helper ---
def _user_history(user_id: int, n: int = 10):
    """
    Return up to n movieIds the user interacted with in training (and optional metadata).
    """
    if USER_TO_INDEX is None or USER_ITEM is None or INDEX_TO_ITEM is None:
        return {"cold_start": True, "train_nnz": 0, "movieIds": [], "items": []}

    if user_id not in USER_TO_INDEX:
        return {"cold_start": True, "train_nnz": 0, "movieIds": [], "items": []}

    uidx = USER_TO_INDEX[user_id]
    row = USER_ITEM[uidx]  # CSR row
    train_nnz = int(row.nnz)

    internal_items = row.indices[: max(0, int(n))].tolist()
    movie_ids: List[int] = []
    items = []

    for iid in internal_items:
        mid = INDEX_TO_ITEM.get(int(iid))
        if mid is None:
            continue
        mid = int(mid)
        movie_ids.append(mid)
        if MOVIE_META is not None and mid in MOVIE_META:
            items.append({"movieId": mid, **MOVIE_META[mid]})
        else:
            items.append({"movieId": mid})

    return {"cold_start": False, "train_nnz": train_nnz, "movieIds": movie_ids, "items": items}


# --- Diverse list overlap exclusion and fill ---
def _exclude_and_fill(user_id: int, candidates: List[int], exclude: set[int], k: int) -> List[int]:
    """
    Take a ranked candidate list, drop anything in `exclude`, and ensure we return exactly k items.
    Fill from POPULAR as needed. If user is known, avoid items already seen in training.
    """
    if k <= 0:
        return []

    out: List[int] = []
    out_set: set[int] = set()

    for mid in candidates:
        mid = int(mid)
        if mid in exclude:
            continue
        if mid in out_set:
            continue
        out.append(mid)
        out_set.add(mid)
        if len(out) >= k:
            return out

    # Fill using popularity, avoiding excludes/duplicates and (if possible) training-seen items.
    seen_internal = set()
    if USER_TO_INDEX is not None and USER_ITEM is not None and ITEM_TO_INDEX is not None and user_id in USER_TO_INDEX:
        uidx = USER_TO_INDEX[user_id]
        row = USER_ITEM[uidx]
        seen_internal = set(row.indices)

    for mid in POPULAR:
        mid = int(mid)
        if mid in exclude or mid in out_set:
            continue
        # Avoid training-seen if we can map it
        if seen_internal and ITEM_TO_INDEX is not None:
            ii = ITEM_TO_INDEX.get(mid)
            if ii is not None and int(ii) in seen_internal:
                continue
        out.append(mid)
        out_set.add(mid)
        if len(out) >= k:
            break

    return out[:k]


@app.on_event("startup")
def startup():
    _load_artifacts()



@app.get("/health")
def health():
    ok = MODEL is not None and USER_ITEM is not None and POPULAR is not None
    return {"ok": ok, "artifact_dir": str(ART_DIR)}


@app.get("/demo")
def demo():
    """Handy demo helper endpoint (no need to remember userIds during interviews)."""
    known_user_id = int(os.getenv("DEMO_KNOWN_USER_ID", "155223"))
    cold_start_user_id = int(os.getenv("DEMO_COLD_START_USER_ID", "999999999"))
    k = int(os.getenv("DEMO_K", "10"))

    return {
        "known_user_id": known_user_id,
        "cold_start_user_id": cold_start_user_id,
        "k": k,
        "examples": {
            "explain_known": f"/explain/{known_user_id}?k={k}&history_n=10",
            "explain_cold": f"/explain/{cold_start_user_id}?k={k}&history_n=10",
            "compare_known": f"/recommend/compare/{known_user_id}?k={k}&include_meta=1",
        },
    }


@app.get("/movies")
def movies(ids: str):
    """
    Lookup MovieLens metadata by comma-separated movieIds.
    Example: /movies?ids=1,2,3
    """
    if MOVIE_META is None:
        raise HTTPException(status_code=500, detail="Movie metadata not loaded (movies.csv missing?)")

    out = []
    for token in ids.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            mid = int(token)
        except ValueError:
            continue

        meta = MOVIE_META.get(mid)
        if meta is None:
            continue

        out.append({"movieId": mid, **meta})

    return {"count": len(out), "movies": out}


@app.get("/stats")
def stats():
    avg_ms = (LAT_TOTAL_MS / LAT_SAMPLES) if LAT_SAMPLES > 0 else 0.0
    return {
        "cache": {
            "ttl_seconds": CACHE_TTL_SECONDS,
            "max_entries": CACHE_MAX_ENTRIES,
            "size": len(REC_CACHE),
            "hits": CACHE_HITS,
            "misses": CACHE_MISSES,
        },
        "latency_ms": {
            "samples": LAT_SAMPLES,
            "avg": avg_ms,
            "max": LAT_MAX_MS,
        },
        "diversity": {
            "on": DIVERSITY_ON,
            "lambda": DIVERSITY_LAMBDA,
            "candidates_mult": DIVERSITY_CANDIDATES_MULT,
            "default_mode": ("diverse" if DIVERSITY_ON else "als"),
        },
    }


@app.get("/recommend/{user_id}")
def recommend_get(user_id: int, k: int = 10, mode: Optional[str] = None):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        m = _normalize_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t0 = time.perf_counter()
    key = (int(user_id), int(k), m)
    cached_hint = key in REC_CACHE
    movie_ids = _recommend(user_id, k, m)
    ms = (time.perf_counter() - t0) * 1000.0
    _record_latency(ms)
    return {"user_id": user_id, "k": k, "mode": m, "movieIds": movie_ids, "latency_ms": ms, "cached_hint": cached_hint}


@app.get("/recommend/compare/{user_id}")
def recommend_compare(user_id: int, k: int = 10, include_meta: int = 0):
    """
    Return ALS vs Diverse recommendations side-by-side for easy A/B demos.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # ALS
    t0 = time.perf_counter()
    als_mode = "als"
    als_key = (int(user_id), int(k), als_mode)
    als_cached = als_key in REC_CACHE
    als_movie_ids = _recommend(user_id, k, als_mode)
    als_ms = (time.perf_counter() - t0) * 1000.0
    _record_latency(als_ms)

    # Diverse (respects global DIVERSITY_ON; normalize in case it's disabled)
    t1 = time.perf_counter()
    diverse_mode = _normalize_mode("diverse")
    # Pull more diverse candidates, then remove ALS items for a clearer A/B demo.
    div_candidates_k = max(int(k) * 5, int(k) + 50)

    # Note: we fetch a larger candidate list for diverse, so the cache key differs.
    # `div_cached` must be computed BEFORE calling `_recommend`, since `_recommend` may populate the cache.
    div_key = (int(user_id), int(div_candidates_k), diverse_mode)
    div_cached = div_key in REC_CACHE

    div_candidates = _recommend(user_id, div_candidates_k, diverse_mode)
    div_movie_ids = _exclude_and_fill(int(user_id), div_candidates, set(als_movie_ids), int(k))

    div_ms = (time.perf_counter() - t1) * 1000.0
    _record_latency(div_ms)

    meta = None
    als_items = None
    diverse_items = None

    # If include_meta=1, include per-list item arrays (frontend-friendly).
    if include_meta and MOVIE_META is not None:
        def _to_items(ids: List[int]):
            out = []
            for mid in ids:
                m = MOVIE_META.get(int(mid))
                if m is None:
                    continue
                out.append({"movieId": int(mid), **m})
            return out

        als_items = _to_items(als_movie_ids)
        diverse_items = _to_items(div_movie_ids)

        # Keep legacy meta dict too (useful for debugging)
        all_ids = list(dict.fromkeys(als_movie_ids + div_movie_ids))
        meta = {mid: MOVIE_META.get(mid) for mid in all_ids if mid in MOVIE_META}

    return {
        "user_id": user_id,
        "k": k,
        "meta": meta,
        "als_items": als_items,
        "diverse_items": diverse_items,
        "als": {
            "mode": als_mode,
            "movieIds": als_movie_ids,
            "latency_ms": als_ms,
            "cached_hint": als_cached,
        },
        "diverse": {
            "mode": diverse_mode,
            "movieIds": div_movie_ids,
            "latency_ms": div_ms,
            "cached_hint": div_cached,
        },
    }


@app.post("/recommend")
def recommend_post(req: RecommendRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        m = _normalize_mode(req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    t0 = time.perf_counter()
    key = (int(req.user_id), int(req.k), m)
    cached_hint = key in REC_CACHE
    movie_ids = _recommend(req.user_id, req.k, m)
    ms = (time.perf_counter() - t0) * 1000.0
    _record_latency(ms)
    return {"user_id": req.user_id, "k": req.k, "mode": m, "movieIds": movie_ids, "latency_ms": ms, "cached_hint": cached_hint}


# --- Minimal transparency endpoint ---
@app.get("/explain/{user_id}")
def explain(user_id: int, k: int = 10, history_n: int = 10, mode: Optional[str] = None):
    """
    Minimal transparency endpoint for demos:
    - shows what the user has seen in training
    - shows current recommendations (ALS vs Diverse)
    """
    if MODEL is None or USER_ITEM is None or POPULAR is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")

    # Mode used for the "primary" list (still returns both ALS and Diverse)
    try:
        primary_mode = _normalize_mode(mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    hist = _user_history(int(user_id), int(history_n))

    als_ids = _recommend(int(user_id), int(k), "als")
    diverse_mode = _normalize_mode("diverse")
    div_candidates_k = max(int(k) * 5, int(k) + 50)
    div_candidates = _recommend(int(user_id), int(div_candidates_k), diverse_mode)
    div_ids = _exclude_and_fill(int(user_id), div_candidates, set(als_ids), int(k))

    als_items = None
    div_items = None
    if MOVIE_META is not None:
        def _to_items(ids: List[int]):
            out = []
            for mid in ids:
                m = MOVIE_META.get(int(mid))
                if m is None:
                    continue
                out.append({"movieId": int(mid), **m})
            return out

        als_items = _to_items(als_ids)
        div_items = _to_items(div_ids)

    return {
        "user_id": int(user_id),
        "k": int(k),
        "mode_requested": (primary_mode if mode is not None else None),
        "mode_used": primary_mode,
        "history": hist,
        "als": {"mode": "als", "movieIds": als_ids, "items": als_items},
        "diverse": {"mode": diverse_mode, "movieIds": div_ids, "items": div_items},
    }