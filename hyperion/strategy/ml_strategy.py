"""
ML多因子选股策略 (Qlib Model + AlphaFusion)
=============================================
基于机器学习预测的多因子选股策略。

核心流程:
  1. 因子提取 (Alpha158)
  2. 模型预测 (LightGBM/Ridge/MLP)
  3. 信号生成 (Top-K选股)
  4. 组合优化 (Risk Budgeting)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from hyperion.strategy.base import BaseStrategy, Signal
from hyperion.alpha.factors import Alpha158
from hyperion.alpha.bayesian import BayesianUpdater

logger = logging.getLogger(__name__)


class MLMultiFactorStrategy(BaseStrategy):
    """机器学习多因子选股策略
    
    用法:
        strategy = MLMultiFactorStrategy(
            symbols=["000001.SZ", "000002.SZ", ...],
            model_type="lightgbm",
            top_k=30,
            rebalance_freq="monthly"
        )
    
    模式:
    - predict: 使用预训练模型预测收益率
    - bayesian: 使用贝叶斯动态权重
    - momentum: 纯动量因子
    """
    
    def __init__(self, name: str = "ml_multi_factor",
                 symbols: Optional[List[str]] = None,
                 model_type: str = "lightgbm",
                 top_k: int = 30,
                 rebalance_freq: str = "monthly",
                 mode: str = "predict",
                 **params):
        super().__init__(name=name, symbols=symbols, params=params)
        self.model_type = model_type
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self.mode = mode
        
        # 因子引擎
        self.alpha_engine = Alpha158()
        self.bayesian: Optional[BayesianUpdater] = None
        
        # 模型
        self._model = None
        self._trained = False
        
        # 调仓计数
        self._bar_count = 0
    
    def on_init(self) -> None:
        super().on_init()
        
        # 初始化贝叶斯更新器
        n_factors = len(self.alpha_engine.feature_names)
        self.bayesian = BayesianUpdater(n_factors=n_factors)
        
        # 初始化模型
        if self.model_type == "lightgbm":
            self._init_lightgbm()
        elif self.model_type == "ridge":
            self._init_ridge()
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using ridge")
            self._init_ridge()
    
    def _init_lightgbm(self):
        """初始化LightGBM模型"""
        try:
            import lightgbm as lgb
            self._model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        except ImportError:
            logger.warning("lightgbm not installed, falling back to ridge")
            self._init_ridge()
    
    def _init_ridge(self):
        """初始化Ridge回归"""
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=1.0)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """训练模型
        
        Args:
            X: 因子DataFrame
            y: 未来收益Series
        """
        if self._model is None:
            self._init_ridge()
        
        # 处理缺失值
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.fillna(0)
        
        # 训练
        self._model.fit(X.values, y.values)
        self._trained = True
        logger.info(f"Model trained on {len(X)} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测收益率"""
        if not self._trained or self._model is None:
            return np.zeros(len(X))
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self._model.predict(X.values)
    
    def on_bar(self, bar_data: Dict[str, pd.Series]) -> List[Signal]:
        """每根K线生成信号"""
        signals = []
        
        # 调仓频率控制
        self._bar_count += 1
        if not self._should_rebalance():
            return signals
        
        # 收集所有标的的特征
        all_features = {}
        for symbol in self.symbols:
            data = self.get_data(symbol, lookback=120)
            if len(data) < 60:
                continue
            
            # 标准化列名
            data_renamed = data.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume"
            })
            if "vwap" not in data_renamed.columns:
                data_renamed["vwap"] = (data_renamed["high"] + data_renamed["low"] + data_renamed["close"]) / 3
            
            # 提取Alpha158因子
            try:
                factors = self.alpha_engine.extract(data_renamed)
                all_features[symbol] = factors.iloc[-1]  # 最新一期因子
            except Exception as e:
                logger.debug(f"Factor extraction failed for {symbol}: {e}")
        
        if not all_features:
            return signals
        
        # 构建因子矩阵
        symbols_list = list(all_features.keys())
        X = pd.DataFrame(all_features).T
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 预测分数
        if self.mode == "predict" and self._trained:
            scores = self.predict(X)
        elif self.mode == "bayesian" and self.bayesian is not None:
            # 使用贝叶斯权重
            weights = self.bayesian.weights
            scores = X.values @ weights
        else:
            # 动量模式: 使用过去20日收益率
            scores = []
            for sym in symbols_list:
                data = self.get_data(sym, lookback=120)
                if len(data) >= 21 and "close" in data.columns:
                    scores.append(data["close"].pct_change(20).iloc[-1])
                else:
                    scores.append(0.0)
            scores = np.array(scores)
        
        # Top-K选股
        top_k = min(self.top_k, len(scores))
        if top_k == 0:
            return signals
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # 生成信号
        total_capital = self.capital
        position_size = total_capital * 0.95 / top_k  # 等权分配 (暂)
        
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            symbol = symbols_list[idx]
            signal = Signal(
                symbol=symbol,
                direction="BUY",
                strength=float(scores[idx] / (np.abs(scores).max() + 1e-12)),
                target_weight=1.0 / top_k
            )
            signals.append(signal)
        
        return signals
    
    def _should_rebalance(self) -> bool:
        """判断是否需要调仓"""
        if self.rebalance_freq == "daily":
            return True
        elif self.rebalance_freq == "weekly":
            return self._bar_count % 5 == 0
        elif self.rebalance_freq == "monthly":
            return self._bar_count % 20 == 0
        return self._bar_count % 20 == 0
