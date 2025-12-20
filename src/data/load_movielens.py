from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-25m"

def load_ratings(nrows: int | None = None):
    path = DATA_DIR / "ratings.csv"
    dtypes = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int32",
    }
    return pd.read_csv(path, dtype=dtypes, nrows=nrows)

def load_movies():
    path = DATA_DIR / "movies.csv"
    return pd.read_csv(path)
