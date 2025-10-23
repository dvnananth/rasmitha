from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from catboost import CatBoostRegressor, Pool


@dataclass
class CatBoostConfig:
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    iterations: int = 500
    loss_function: str = "RMSE"
    random_seed: int = 42
    verbose: int = 0


class CatBoostSpecialist:
    """CatBoost model to predict residual r_total from context features.

    Typically trained on the dataset's attn_context features (latest step features).
    """

    def __init__(self, config: CatBoostConfig = CatBoostConfig()):
        self.config = config
        self.model = CatBoostRegressor(
            depth=config.depth,
            learning_rate=config.learning_rate,
            l2_leaf_reg=config.l2_leaf_reg,
            iterations=config.iterations,
            loss_function=config.loss_function,
            random_seed=config.random_seed,
            verbose=config.verbose,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        train_pool = Pool(X, y)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)
        self.model.fit(train_pool, eval_set=eval_set)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("CatBoostSpecialist must be fitted before predicting.")
        return self.model.predict(X).reshape(-1, 1)

    def is_fitted(self) -> bool:
        return self._fitted
