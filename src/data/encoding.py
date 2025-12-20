import pandas as pd

def build_id_mappings(
    interactions: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
):
    """
    Build 0-based contiguous integer ids for users and items
    based on the interactions in *train* data.
    """
    unique_users = interactions[user_col].unique()
    unique_items = interactions[item_col].unique()

    user_to_index = {u: i for i, u in enumerate(unique_users)}
    index_to_user = {i: u for u, i in user_to_index.items()}

    item_to_index = {m: i for i, m in enumerate(unique_items)}
    index_to_item = {i: m for m, i in item_to_index.items()}

    return user_to_index, index_to_user, item_to_index, index_to_item
