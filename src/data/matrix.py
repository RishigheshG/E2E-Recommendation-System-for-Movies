from typing import Literal
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def build_interaction_matrix(
    interactions: pd.DataFrame,
    user_to_index: dict,
    item_to_index: dict,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    use_ratings: bool = True,
    alpha: float = 40.0,
    orientation: Literal["user_item", "item_user"] = "user_item",
):
    """
    Build a sparse interaction matrix for implicit ALS (default: user×item).

    - user_to_index, item_to_index: mappings from raw IDs to 0-based indices
    - use_ratings:
        If True: confidence = 1 + alpha * rating
        If False: confidence = 1 for all interactions (pure implicit)

    Returns:
      CSR matrix of shape:
        - (num_users, num_items) if orientation="user_item" (default)
        - (num_items, num_users) if orientation="item_user"
    """
    user_indices = interactions[user_col].map(user_to_index).values
    item_indices = interactions[item_col].map(item_to_index).values

    if use_ratings:
        # Treat ratings as confidence: larger rating -> higher confidence
        data = 1.0 + alpha * interactions[rating_col].values.astype(np.float32)
    else:
        data = np.ones(len(interactions), dtype=np.float32)

    if orientation == "user_item":
        # rows = users, cols = items
        mat = coo_matrix(
            (data, (user_indices, item_indices)),
            shape=(len(user_to_index), len(item_to_index)),
            dtype=np.float32,
        )
    elif orientation == "item_user":
        # rows = items, cols = users
        mat = coo_matrix(
            (data, (item_indices, user_indices)),
            shape=(len(item_to_index), len(user_to_index)),
            dtype=np.float32,
        )
    else:
        raise ValueError('orientation must be "user_item" or "item_user"')

    return mat.tocsr()