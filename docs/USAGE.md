# Hyperion+ v27 使用说明

## 快速入门

### 1. 安装

```bash
git clone <your-repo-url>
cd quant_trading
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -m tests.v27_ultra_benchmark
```

预期输出: 7/7 PASS。

## 基础使用

### 场景 1: 单因子提取

```python
import pandas as pd
from hyperion.alpha.alpha_ultra import AlphaUltra, FactorConfig

# 加载数据 (AKShare/Tushare/CSV)
df = pd.read_csv("data/000300.SH.csv")  # 需要包含 OHLCV 列

# 提取因子
config = FactorConfig(use_alpha158=True, use_alpha360=True)
engine = AlphaUltra(config)
factors = engine.extract(df)  # 返回 DataFrame (698+ 列)

print(factors.head())
```

### 场景 2: 训练模型并回测

```python
from hyperion import HyperionEngine
from hyperion.ultra_orchestrator import OrchestratorConfig

# 配置
config = OrchestratorConfig(
    symbols=["000300.SH"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    model_names=["lgb", "xgb", "stacking"],
    backtest_mode="event_driven",
)

# 初始化
engine = HyperionEngine(config)

# 训练
engine.train()

# 回测
results = engine.backtest()
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Return: {results['total_return']:.2f}%")
```

### 场景 3: 高频微观结构信号提取

```python
from hyperion.hft.hf_engine import MicrostructureAlpha

# 从 Tick 数据提取微观结构 Alpha
ms_alpha = MicrostructureAlpha()
for tick in tick_stream:
    features = ms_alpha.process_tick(tick)
    print(features)  # 输出订单簿特征/资金流/价格冲击
```

### 场景 4: 组合优化

```python
import numpy as np
from hyperion.portfolio.ultra_optimizer import OptimizerFactory

n_assets = 5
expected = np.random.randn(n_assets) * 0.01
cov = np.diag(np.abs(np.random.randn(n_assets) * 0.01))

# 风险预算优化
opt = OptimizerFactory.create("risk_budgeting")
# 层次风险平衡
opt = OptimizerFactory.create("hrp")
# 均值-CVaR
opt = OptimizerFactory.create("mean_cvar")

weights = opt.optimize(expected, cov)
print(f"最优权重: {weights}")
```

### 场景 5: 在线学习 (自适应市场变化)

```python
from hyperion.online.ultra_online import OnlineLearningPipeline

# 初始化
pipeline = OnlineLearningPipeline(models={"lgb": lgb_model})

# 日线更新
for date in trading_dates:
    prediction = models["lgb"].predict(features.loc[date])
    needs_retrain, weights = pipeline.update(
        date, {"lgb": prediction}, actual_returns, features
    )
    if needs_retrain:
        new_model = pipeline.retrain(data, target, feature_cols)
```

## 高级用法

### 自定义模型集成

```python
from hyperion.model_zoo.ultra_models import BaseModel

class MyCustomModel(BaseModel):
    def fit(self, X, y):
        # 自定义训练逻辑
        pass
    
    def predict(self, X):
        # 自定义预测逻辑
        pass

# 注册
ModelFactory._REGISTRY["my_model"] = MyCustomModel
```

### 自定义策略

```python
from hyperion.engine.ultra_backtest import Strategy, Order, OrderSide, OrderType

class MyStrategy(Strategy):
    def on_bar(self, bar):
        # 每根K线触发
        if bar["close"] > bar["ma20"]:
            return [Order(
                id="001", symbol="000300.SH",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=100
            )]
```

### 自定义因子

```python
from hyperion.alpha.alpha_ultra import AlphaUltra

class MyAlpha(AlphaUltra):
    def _custom_features(self, df):
        return {"MYFACTOR": df["close"] / df["open"]}
    
    def extract(self, df):
        features = super().extract(df)
        features.update(self._custom_features(df))
        return features
```

## 常见问题

### Q: 运行时报 `No module named 'scipy'`?

A: `pip install scipy` 或使用 `pip install -r requirements.txt` 安装全部依赖。

### Q: LightGBM/XGBoost 安装失败?

A: 确保编译器可用：
- macOS: `brew install libomp`
- Ubuntu: `apt-get install libomp-dev`

### Q: PyTorch 模型 (NeuralSDE/GNN/TFT) 无法使用?

A: `pip install torch torch-geometric pytorch-forecasting`，这些是可选依赖。

### Q: 能否用于实盘交易?

A: 可以。通过 `UltraOrchestrator` 的 `execute()` 方法，支持 Paper Trading 和 Live Trading。

## 性能优化

### 并行计算

```python
from hyperion.model_zoo.ultra_models import ModelFactory

# 多模型并行benchmark
results = ModelFactory.benchmark_all(X_train, y_train, X_test, y_test)
```

### 因子缓存

```python
from hyperion.alpha.alpha_ultra import AlphaUltra
# AlphaUltra 已内置 Parquet 缓存
factors = engine.extract(df, use_cache=True)
```

## 联系方式

- 项目: Hyperion Quant v27
- 问题: 在 GitHub Issues 中提交

## 许可

Proprietary — 内部使用
