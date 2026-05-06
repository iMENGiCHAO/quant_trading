"""
Model Zoo Base — ABC and Registry for all models.
"""
from __future__ import annotations

import abc
from typing import Dict, Type, Optional, Any, List
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class BaseModel(abc.ABC):
    """Abstract base for all prediction models."""

    model_type: str = "base"

    def __init__(self, **kwargs):
        self._model: Any = None
        self._fitted = False
        self._params = kwargs

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[List] = None):
        """Train the model."""
        pass

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        pass

    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame) -> np.ndarray:
        """Fit and predict."""
        self.fit(X_train, y_train)
        return self.predict(X_test)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_params(self) -> Dict[str, Any]:
        return self._params

    def __repr__(self):
        return f"{self.__class__.__name__}(fitted={self._fitted})"


class ModelRegistry:
    """Registry for all available models."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type[BaseModel]):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        return cls.get(name)(**kwargs)