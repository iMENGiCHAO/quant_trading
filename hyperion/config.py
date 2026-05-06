"""
Hyperion Configuration System
===============================
Pydantic-based configuration inspired by FinRL-X.
Supports YAML file + environment variable overrides.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class DataSourceType(str, Enum):
    AKSHARE = "akshare"
    TUSHARE = "tushare"
    BAOSTOCK = "baostock"
    CSV = "csv"
    SQLITE = "sqlite"


class BrokerType(str, Enum):
    PAPER = "paper"
    CTP = "ctp"
    XTP = "xtp"
    IB = "ib"


class ObjectiveType(str, Enum):
    SHARPE = "sharpe"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR = "calmar"
    SORTINO = "sortino"
    RETURN = "return"
    MULTI = "multi"


@dataclass
class DataConfig:
    """数据层配置 (Qlib DataServer + Freqtrade data management)"""
    source: DataSourceType = DataSourceType.AKSHARE
    db_path: str = "~/.quant_trading/ashare.db"
    cache_dir: str = "~/.quant_trading/cache"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    symbols: List[str] = field(default_factory=lambda: ["000300.SH", "000905.SH"])
    fields: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "amount", "turnover"
    ])
    # Qlib-style Parquet cache
    use_parquet_cache: bool = True
    # Point-in-time 无前视偏差隔离天数
    purge_days: int = 5


@dataclass
class AlphaConfig:
    """因子层配置 (Qlib Alpha158 + 贝叶斯 + 因果)"""
    # Alpha158 因子开关
    use_alpha158: bool = True
    use_alpha360: bool = False
    # 贝叶斯在线学习
    use_bayesian: bool = True
    bayesian_window: int = 60
    # 因果发现
    use_causal: bool = False
    # 技术指标
    indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bbands", "atr", "sma", "ema"
    ])
    # IC筛选阈值
    min_ic: float = 0.02
    max_correlation: float = 0.7


@dataclass
class StrategyConfig:
    """策略层配置"""
    name: str = "ml_multi_factor"
    # 预测目标 (Qlib-style)
    label: str = "forward_return_5d"
    # 调仓频率
    rebalance_freq: str = "monthly"  # daily, weekly, monthly
    # 持仓数量
    max_positions: int = 30
    # 训练相关
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class EngineConfig:
    """引擎层配置"""
    mode: str = "backtest"  # backtest, paper, live
    initial_capital: float = 1_000_000.0
    # 回测参数 (Backtrader style)
    commission: float = 0.0003  # 万三佣金
    slippage: float = 0.001    # 0.1% 滑点
    stamp_duty: float = 0.001   # 千一印花税(卖出)
    # A股约束
    t_plus_1: bool = True       # T+1交易
    price_limit: float = 0.10   # 涨跌停限制
    st_filter: bool = True      # ST过滤


@dataclass
class RiskConfig:
    """风控层配置 (VnPy risk manager + 组合优化)"""
    # 仓位管理
    max_position_pct: float = 0.10   # 单一标的最大仓位10%
    max_sector_pct: float = 0.30     # 单一行业最大仓位30%
    max_total_exposure: float = 0.95 # 最大总仓位95%
    # 止损
    stop_loss_pct: float = 0.05      # 单笔止损5%
    daily_loss_limit: float = 0.03   # 日内最大亏损3%
    # 组合优化方法
    optimization_method: str = "risk_budgeting"  # hrp, mean_variance, risk_budgeting


@dataclass
class ExecutionConfig:
    """执行层配置 (Hummingbot style broker separation)"""
    broker: BrokerType = BrokerType.PAPER
    # 算法交易 (VnPy algo trading)
    algo: str = "twap"  # twap, vwap, iceberg, sniper
    order_timeout: int = 300  # 订单超时秒数


@dataclass
class HyperoptConfig:
    """超参优化配置 (Freqtrade Hyperopt + Optuna)"""
    objective: ObjectiveType = ObjectiveType.SHARPE
    max_evals: int = 500
    cv_folds: int = 5
    search_space: str = "strategy_params"
    sampler: str = "tpe"  # tpe, random, grid
    pruner: str = "median"
    n_jobs: int = 1
    # 多目标优化
    multi_objectives: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """分析层配置 (Backtrader Analyzer + VectorBT metrics)"""
    benchmark: str = "000300.SH"  # 基准指数
    risk_free_rate: float = 0.03  # 无风险利率
    # 报告格式
    report_format: str = "html"   # html, pdf, json
    plot: bool = True


@dataclass
class HyperionConfig:
    """Hyperion 主配置 (融合所有层)"""
    data: DataConfig = field(default_factory=DataConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    hyperopt: HyperoptConfig = field(default_factory=HyperoptConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "HyperionConfig":
        """从YAML文件加载配置"""
        if yaml is None:
            raise ImportError("pyyaml is required for YAML config. Install with: pip install pyyaml")
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict, prefix: str = "") -> Any:
        """递归构建嵌套dataclass"""
        if prefix == "":
            data = DataConfig(**d.get("data", {}))
            alpha = AlphaConfig(**d.get("alpha", {}))
            strategy = StrategyConfig(**d.get("strategy", {}))
            engine = EngineConfig(**d.get("engine", {}))
            risk = RiskConfig(**d.get("risk", {}))
            execution = ExecutionConfig(**d.get("execution", {}))
            hyperopt = HyperoptConfig(**d.get("hyperopt", {}))
            analysis = AnalysisConfig(**d.get("analysis", {}))
            return cls(data=data, alpha=alpha, strategy=strategy, engine=engine,
                      risk=risk, execution=execution, hyperopt=hyperopt, analysis=analysis)
        return d

    def to_yaml(self, path: str) -> None:
        """保存配置到YAML"""
        if yaml is None:
            raise ImportError("pyyaml is required for YAML config. Install with: pip install pyyaml")
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# Global config instance
config: Optional[HyperionConfig] = None


def load_config(config_path: Optional[str] = None) -> HyperionConfig:
    """加载全局配置"""
    global config
    if config_path is None:
        config_path = os.environ.get("HYPERION_CONFIG", "config.yaml")

    if os.path.exists(config_path):
        config = HyperionConfig.from_yaml(config_path)
    else:
        config = HyperionConfig()  # defaults

    return config


def get_config() -> HyperionConfig:
    """获取全局配置"""
    global config
    if config is None:
        config = load_config()
    return config
