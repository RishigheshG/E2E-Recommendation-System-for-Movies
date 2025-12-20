from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from data.load_movielens import load_ratings, load_movies
from models.popularity import PopularityRecommender

ratings = load_ratings()
movies = load_movies()

pop_model = PopularityRecommender(top_k=10)
pop_model.fit(ratings)

test_user = ratings['userId'].iloc[0]
recs = pop_model.recommend(test_user, ratings)
print("User:", test_user)
print("Recommended movieIds:", recs)

# Show titles for sanity
df_movies = movies.set_index('movieId')
titles = [df_movies.loc[movie_id, 'title'] if movie_id in df_movies.index else None for movie_id in recs]
print("Recommended titles:", titles)
