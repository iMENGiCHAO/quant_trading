"""
Model Trainer — Unified training pipeline with validation, early stopping, and logging.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from hyperion.model_zoo.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration."""
    epochs: int = 100
    early_stopping: int = 10
    cv_folds: int = 5
    validation_ratio: float = 0.2
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    device: str = "cpu"
    verbose: bool = True


class ModelTrainer:
    """Unified model training pipeline."""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()

    def train(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train a model and return metrics.

        Args:
            model: Model instance
            X: Feature dataframe
            y: Target series
            eval_set: Optional validation data [(X_val, y_val)]
            sample_weight: Optional sample weights

        Returns:
            Training metrics dict
        """
        import time
        start = time.time()

        model.fit(X, y, eval_set=eval_set)

        metrics = {
            "train_time": time.time() - start,
            "fitted": model.fitted,
        }

        if self.config.verbose:
            logger.info(f"Training complete: {model} in {metrics['train_time']:.1f}s")

        return metrics

    def cross_validate(
        self,
        model_factory: Callable[[], BaseModel],
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
    ) -> List[Dict[str, float]]:
        """Cross-validate a model factory."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        tscv = TimeSeriesSplit(n_splits=n_folds)
        results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            fold_metrics = {
                "fold": fold,
                "mse": mean_squared_error(y_val, preds),
                "mae": mean_absolute_error(y_val, preds),
                "corr": np.corrcoef(y_val, preds)[0, 1] if len(preds) > 1 else 0,
            }
            results.append(fold_metrics)

        return results