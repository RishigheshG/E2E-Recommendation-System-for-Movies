import pandas as pd

class PopularityRecommender:
    def __init__(self, top_k: int = 10, user_col: str = "userId", item_col: str = "movieId"):
        self.top_k = top_k
        self.user_col = user_col
        self.item_col = item_col
        self.popular_items = None

    def fit(self, interactions: pd.DataFrame):
        """
        interactions: train set with at least [user_col, item_col]
        """
        item_counts = (
            interactions
            .groupby(self.item_col)[self.user_col]
            .count()
            .sort_values(ascending=False)
        )

        self.popular_items = item_counts.index.tolist()
        self._train_interactions = interactions[[self.user_col, self.item_col]]
        return self

    def recommend(self, user_id: int) -> list:
        """
        Recommend top_k items for user_id, excluding already seen items (in train).
        """
        if self.popular_items is None:
            raise ValueError("Model not fitted. Call fit() first.")

        user_interactions = self._train_interactions
        seen_items = set(
            user_interactions.loc[user_interactions[self.user_col] == user_id, self.item_col]
        )

        recs = [item for item in self.popular_items if item not in seen_items]
        return recs[: self.top_k]
