from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_movielens_dir() -> Path:
    """
    Return the directory containing the MovieLens 25M CSV files.

    Priority:
      1) Environment variable MOVIELENS_25M_DIR
      2) <project_root>/data/raw/ml-25m
    """
    env = os.getenv("MOVIELENS_25M_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "data" / "raw" / "ml-25m").resolve()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Fix:\n"
            "  - Put MovieLens 25M under <project>/data/raw/ml-25m, OR\n"
            "  - Set MOVIELENS_25M_DIR to the folder containing ratings.csv/movies.csv\n"
            "Example:\n"
            "  export MOVIELENS_25M_DIR='/absolute/path/to/ml-25m'\n"
        )


def load_ratings(
    nrows: int | None = None,
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load ratings.csv with tight dtypes for speed/memory.

    Tip: pass usecols=['userId','movieId','rating','timestamp'] (default)
    """
    data_dir = get_movielens_dir()
    path = data_dir / "ratings.csv"
    _require_file(path)

    default_cols = ["userId", "movieId", "rating", "timestamp"]
    cols = list(usecols) if usecols is not None else default_cols

    dtypes = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int32",
    }
    return pd.read_csv(path, dtype=dtypes, nrows=nrows, usecols=cols)


def load_movies(
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load movies.csv (movieId, title, genres).
    """
    data_dir = get_movielens_dir()
    path = data_dir / "movies.csv"
    _require_file(path)

    return pd.read_csv(path, usecols=usecols)


def load_links(
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load links.csv (movieId, imdbId, tmdbId).
    Useful for web/mobile demos that show posters via TMDB later.
    """
    data_dir = get_movielens_dir()
    path = data_dir / "links.csv"
    _require_file(path)

    # imdbId/tmdbId can be missing -> use pandas nullable Int64 if you select those columns
    return pd.read_csv(path, usecols=usecols)
