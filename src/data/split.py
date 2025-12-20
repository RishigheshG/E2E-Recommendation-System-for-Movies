import pandas as pd

def temporal_train_val_test_split(
    interactions: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    time_col: str = "timestamp",
    min_interactions: int = 3,
):
    """
    Per-user temporal split:
      - Users with < min_interactions: all interactions -> train only
      - Users with >= min_interactions:
          oldest ... second-last-1 -> train
          second-last -> val
          last -> test
    """
    # Sort by user and time
    interactions = interactions.sort_values([user_col, time_col])

    train_parts = []
    val_parts = []
    test_parts = []

    for user_id, user_df in interactions.groupby(user_col):
        n = len(user_df)
        if n < min_interactions:
            # Not enough history: use only for training
            train_parts.append(user_df)
            continue

        # iloc indexing: up to n-2 -> train, n-2 -> val, n-1 -> test
        train_parts.append(user_df.iloc[:-2])
        val_parts.append(user_df.iloc[[-2]])
        test_parts.append(user_df.iloc[[-1]])

    train_df = pd.concat(train_parts, ignore_index=True)

    if val_parts:
        val_df = pd.concat(val_parts, ignore_index=True)
    else:
        val_df = pd.DataFrame(columns=interactions.columns)

    if test_parts:
        test_df = pd.concat(test_parts, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=interactions.columns)

    return train_df, val_df, test_df
