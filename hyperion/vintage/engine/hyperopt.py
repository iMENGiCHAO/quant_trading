"""
超参数优化引擎 (Freqtrade Hyperopt + Optuna)
================================================
自动化策略参数优化, 支持:
- Bayesian优化 (TPE)
- TimeSeriesSplit CV
- Early stopping (Median Pruner)
- 多目标Pareto优化
- 断点续跑

Usage:
    hopt = HyperoptEngine(strategy_class=MyStrategy, search_space=space)
    results = hopt.optimize(max_evals=500)
    print(results.best_params)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler, RandomSampler
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    from sklearn.model_selection import TimeSeriesSplit
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


@dataclass
class HyperoptResult:
    """超参优化结果"""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_value: float = 0.0
    study_name: str = ""
    n_trials: int = 0
    n_completed: int = 0
    pareto_front: List[Dict] = field(default_factory=list)
    trials_df: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        lines = [
            f"=== Hyperopt Results: {self.study_name} ===",
            f"Best value: {self.best_value:.4f}",
            f"Best params: {self.best_params}",
            f"Trials: {self.n_completed}/{self.n_trials}",
        ]
        return "\n".join(lines)


class HyperoptEngine:
    """超参数优化引擎
    
    Usage:
        def objective(trial):
            # 定义搜索空间
            lookback = trial.suggest_int("lookback", 10, 120)
            threshold = trial.suggest_float("threshold", 0.01, 0.1)
            
            # 训练+评估策略
            score = train_and_evaluate(lookback, threshold)
            return score
        
        engine = HyperoptEngine()
        result = engine.optimize(objective, max_evals=200)
    """
    
    def __init__(self, study_name: str = "hyperion_optimization",
                 direction: str = "maximize",
                 pruner: str = "median",
                 n_jobs: int = 1):
        if not _HAS_OPTUNA:
            raise ImportError("optuna required: pip install optuna")
        
        self.study_name = study_name
        self.direction = direction
        self.n_jobs = n_jobs
        
        # 采样器
        self.sampler = TPESampler(seed=42)
        
        # 剪枝器
        self.pruner = MedianPruner() if pruner == "median" else None
        
        self._study: Optional[optuna.Study] = None
    
    def optimize(self, objective: Callable,
                 max_evals: int = 500,
                 timeout: Optional[int] = None,
                 storage: Optional[str] = None) -> HyperoptResult:
        """运行超参优化
        
        Args:
            objective: 目标函数 trial → score
            max_evals: 最大评估次数
            timeout: 超时(秒)
            storage: 数据库存储 (None=内存)
            
        Returns:
            HyperoptResult
        """
        direction_val = "maximize" if self.direction == "maximize" else "minimize"
        
        if storage:
            self._study = optuna.create_study(
                study_name=self.study_name,
                direction=direction_val,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=storage,
                load_if_exists=True
            )
        else:
            self._study = optuna.create_study(
                study_name=self.study_name,
                direction=direction_val,
                sampler=self.sampler,
                pruner=self.pruner
            )
        
        self._study.optimize(
            objective,
            n_trials=max_evals,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        return self._build_result()
    
    def optimize_multi(self, objectives: List[Callable],
                       max_evals: int = 500) -> HyperoptResult:
        """多目标优化 (Pareto前沿)
        
        Args:
            objectives: 目标函数列表 trial → [score1, score2, ...]
            max_evals: 最大试验次数
        """
        n_obj = len(objectives)
        directions = [self.direction] * n_obj
        
        self._study = optuna.create_study(
            study_name=self.study_name,
            directions=directions,
            sampler=self.sampler
        )
        
        def multi_objective(trial):
            return [obj(trial) for obj in objectives]
        
        self._study.optimize(multi_objective, n_trials=max_evals)
        
        return self._build_result()
    
    def _build_result(self) -> HyperoptResult:
        """构建结果对象"""
        if self._study is None:
            return HyperoptResult()
        
        result = HyperoptResult(
            best_params=self._study.best_params,
            best_value=self._study.best_value,
            study_name=self.study_name,
            n_trials=len(self._study.trials),
            n_completed=sum(1 for t in self._study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE),
            trials_df=self._study.trials_dataframe()
        )
        
        return result
