from typing import List, Dict, Optional

import implicit
from scipy.sparse import csr_matrix


class ALSRecommender:
    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        use_gpu: bool = False,
        k: int = 10,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.k = k

        self.model = None

        self.user_to_index: Optional[Dict[int, int]] = None
        self.index_to_user: Optional[Dict[int, int]] = None
        self.item_to_index: Optional[Dict[int, int]] = None
        self.index_to_item: Optional[Dict[int, int]] = None

        # Dense list mapping internal item index -> raw movieId
        self.index_to_item_list: Optional[List[int]] = None

        # Matrices
        self.user_item_csr: Optional[csr_matrix] = None  # user x item (CSR) for training + recommending
        self.item_user_csr: Optional[csr_matrix] = None  # item x user (CSR) reference/optional

    def fit(
        self,
        user_item_matrix: csr_matrix = None,
        user_to_index: Dict[int, int] = None,
        index_to_user: Dict[int, int] = None,
        item_to_index: Dict[int, int] = None,
        index_to_item: Dict[int, int] = None,
        # Backward-compat keyword
        item_user_matrix: csr_matrix = None,
    ):
        # Backward compatibility: older callers may pass item_user_matrix
        if user_item_matrix is None and item_user_matrix is not None:
            user_item_matrix = item_user_matrix

        if user_item_matrix is None:
            raise ValueError("fit() requires user_item_matrix (user×item CSR matrix).")

        if user_to_index is None or index_to_user is None or item_to_index is None or index_to_item is None:
            raise ValueError("fit() requires user and item index mappings.")

        # Sanity checks: incoming matrix must be user×item and match mapping sizes
        expected_users = len(user_to_index)
        expected_items = len(item_to_index)
        if user_item_matrix.shape != (expected_users, expected_items):
            raise ValueError(
                f"user_item_matrix shape {user_item_matrix.shape} does not match "
                f"(n_users, n_items)=({expected_users}, {expected_items}). "
                "Your mappings and matrix construction are inconsistent."
            )

        self.user_to_index = user_to_index
        self.index_to_user = index_to_user
        self.item_to_index = item_to_index
        self.index_to_item = index_to_item

        # Build dense list mapping internal item index -> raw itemId
        n_items = len(item_to_index)
        self.index_to_item_list = [None] * n_items
        for raw_item_id, idx in item_to_index.items():
            self.index_to_item_list[int(idx)] = raw_item_id

        # Ensure CSR formats
        self.user_item_csr = user_item_matrix.tocsr()
        self.item_user_csr = self.user_item_csr.T.tocsr()

        if self.use_gpu:
            self.model = implicit.gpu.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
            )
        else:
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
            )

        # Train on user×item
        self.model.fit(self.user_item_csr)

        # ---- Post-fit sanity checks ----
        if self.user_item_csr.shape[0] != self.model.user_factors.shape[0]:
            raise ValueError(
                f"user_item_csr rows ({self.user_item_csr.shape[0]}) != "
                f"model.user_factors rows ({self.model.user_factors.shape[0]}). "
                "Your user mapping / matrix orientation is inconsistent."
            )

        if self.user_item_csr.shape[1] != self.model.item_factors.shape[0]:
            raise ValueError(
                f"user_item_csr cols ({self.user_item_csr.shape[1]}) != "
                f"model.item_factors rows ({self.model.item_factors.shape[0]}). "
                "Your item mapping / matrix orientation is inconsistent."
            )

        return self

    def recommend(self, user_raw_id: int) -> List[int]:
        if self.model is None or self.user_to_index is None or self.user_item_csr is None:
            raise ValueError("Model not fitted.")

        if user_raw_id not in self.user_to_index:
            return []

        user_idx = int(self.user_to_index[user_raw_id])

        if user_idx < 0 or user_idx >= self.user_item_csr.shape[0]:
            return []

        # Work around implicit's strict user_items shape check by passing a single-row
        # user×item CSR matrix for this user and asking implicit to recalculate the
        # user's factors from this row.
        user_row = self.user_item_csr[user_idx]  # shape: (1, n_items), CSR

        item_indices, _ = self.model.recommend(
            0,
            user_row,
            N=self.k,
            recalculate_user=True,
        )

        # Filter out items the user has already interacted with in training
        seen_items = set(user_row.indices)

        mapped: List[int] = []
        n_items = len(self.index_to_item_list) if self.index_to_item_list is not None else 0
        for i in item_indices:
            ii = int(i)
            if ii in seen_items:
                continue
            if 0 <= ii < n_items:
                raw = self.index_to_item_list[ii]
                if raw is not None:
                    mapped.append(raw)
        return mapped