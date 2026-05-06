"""
GBDT Models — LightGBM, XGBoost, CatBoost with unified interface.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Any
import numpy as np
import pandas as pd

from hyperion.model_zoo.base import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM model with early stopping and feature importance."""

    model_type = "lightgbm"

    def __init__(self, objective: str = "regression", metric: str = "rmse",
                 num_leaves: int = 31, learning_rate: float = 0.05,
                 n_estimators: int = 500, early_stopping: int = 20,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0, reg_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        try:
            import lightgbm as lgb
            self._lgb = lgb
        except ImportError:
            raise ImportError("lightgbm required. Install: pip install lightgbm")

        self.objective = objective
        self.metric = metric
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

    def _init_model(self):
        return self._lgb.LGBMRegressor(
            objective=self.objective,
            metric=self.metric,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            verbose=-1,
            random_state=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        self._model = self._init_model()
        if eval_set:
            self._model.fit(X, y, eval_set=eval_set,
                          eval_metric=self.metric,
                          callbacks=[self._lgb.early_stopping(self.early_stopping),
                                     self._lgb.log_evaluation(0)])
        else:
            self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self._model.feature_importances_,
                          index=self._model.feature_name_)


@ModelRegistry.register("lightgbm")
class LightGBMModelReg(LightGBMModel):
    pass


class XGBoostModel(BaseModel):
    """XGBoost model."""

    model_type = "xgboost"

    def __init__(self, objective: str = "reg:squarederror",
                 learning_rate: float = 0.05, n_estimators: int = 500,
                 max_depth: int = 6, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, early_stopping: int = 20, **kwargs):
        super().__init__(**kwargs)
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError("xgboost required. Install: pip install xgboost")

        self.objective = objective
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping = early_stopping

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        params = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": 42,
            "verbosity": 0,
        }
        self._model = self._xgb.XGBRegressor(**params, n_estimators=self.n_estimators)
        if eval_set:
            self._model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


@ModelRegistry.register("xgboost")
class XGBoostModelReg(XGBoostModel):
    pass


class CatBoostModel(BaseModel):
    """CatBoost model."""

    model_type = "catboost"

    def __init__(self, learning_rate: float = 0.05, iterations: int = 500,
                 depth: int = 6, l2_leaf_reg: float = 3.0,
                 early_stopping: int = 20, **kwargs):
        super().__init__(**kwargs)
        try:
            import catboost as cb
            self._cb = cb
        except ImportError:
            raise ImportError("catboost required. Install: pip install catboost")

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping = early_stopping

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        self._model = self._cb.CatBoostRegressor(
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=42,
            verbose=False,
        )
        if eval_set:
            self._model.fit(X, y, eval_set=eval_set,
                          early_stopping_rounds=self.early_stopping, verbose=False)
        else:
            self._model.fit(X, y, verbose=False)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


@ModelRegistry.register("catboost")
class CatBoostModelReg(CatBoostModel):
    pass