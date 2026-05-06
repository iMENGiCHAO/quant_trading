# Hyperion+ v27 开发文档

## 目录

- [项目概览](#项目概览)
- [架构设计](#架构设计)
- [模块说明](#模块说明)
- [开发环境](#开发环境)
- [构建与测试](#构建与测试)
- [API参考](#api参考)
- [贡献指南](#贡献指南)

## 项目概览

Hyperion+ v27 是 Alpha Hunter 量化交易系统的下一代旗舰版本，全面超越 Microsoft QLib。

### 设计哲学

> "全盘吸收 QLib 精华，指数级增强。
> 没有任何功能缩减，只有功能超越。"

### 核心指标

| 指标 | Hyperion+ | Microsoft QLib |
|------|-----------|---------------|
| 因子数 | 698+ | 360 |
| 模型数 | 14+ (含前沿) | 40+ (传统) |
| 高频/微观结构 | ✅ | ❌ |
| 事件驱动回测 | ✅ | ❌ |
| 在线学习 | ✅ | ❌ |
| 全链路自动化 | ✅ | ❌ |

## 架构设计

```
Layer 0: 人机交互层 (CLI/Gradio/API)
  ↑
Layer 1: 编排层 (UltraOrchestrator)
  → 多策略并行、A/B测试、动态权重
  ↑
Layer 2: Alpha引擎层 (AlphaUltra)
  → 798因子 = Alpha158 + Alpha360 + HF100 + ALT50 + CAU30
  ↑
Layer 3: 预测模型层 (ModelZoo Ultra)
  → 14+ 模型: LightGBM/XGB/RF + Stacking + NeuralSDE + TFT + GNN + RL
  → 动态选择: 根据市场状态自动切换
  ↑
Layer 4: 组合优化层 (PortfolioOptimizer)
  → RiskBudgeting + HRP + Mean-CVaR + OnlineAdaptive
  ↑
Layer 5: 执行层 (EventBacktest + HF Engine)
  → 事件驱动回测 + 高频撮合引擎
  → 算法交易: TWAP/VWAP/Sniper/Iceberg
  → 实时风控: 止损/仓位/极端场景
```

## 模块说明

### AlphaUltra (hyperion/alpha/alpha_ultra.py)

超级因子引擎，698+ 量价因子。

```python
from hyperion.alpha.alpha_ultra import AlphaUltra, FactorConfig

config = FactorConfig(use_alpha158=True, use_alpha360=True, use_hf_factors=True)
engine = AlphaUltra(config)
factors = engine.extract(df)  # df: OHLCV DataFrame
```

**关键特性:**
- Alpha158: Microsoft QLib 原版158个因子 (完整保留)
- Alpha360扩展: 扩展窗口、加权收益率、对数收益率 (新增)
- HF100: 高频微观结构因子 (新增，QLib无)
- ALT50: 另类数据融合因子 (新增，QLib无)
- CAU30: 因果推理因子 (新增，QLib无)
- 共698+因子

### ModelZoo Ultra (hyperion/model_zoo/ultra_models.py)

统一模型工厂，支持一键创建/基准测试/动态选择。

```python
from hyperion.model_zoo.ultra_models import ModelFactory

# 创建模型
model = ModelFactory.create("lgb")
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 全模型benchmark
results = ModelFactory.benchmark_all(X_train, y_train, X_test, y_test)
```

**模型列表:**
- 传统ML: lgb, xgb, rf, lasso, ridge, elasticnet, linear, gbdt, ar
- 集成: stacking (6层基模型 + 1层元模型)
- 前沿: neuralsde, tft, gnn, rl_portfolio
- 动态选择: 根据市场状态 (Bull/Bear/HighVol) 自动切换

### HF Engine (hyperion/hft/hf_engine.py)

高频/微观结构引擎。QLib 无此模块。

```python
from hyperion.hft.hf_engine import MicrostructureAlpha, OrderBook

# 订单簿重建
ob = OrderBook(max_depth=10)
ob.update_lob(timestamp, bid_levels, ask_levels)

# 冰山检测
iceberg = IcebergDetector()

# 狙击引擎
sniper = SniperEngine()
signal = sniper.evaluate(order_book, tick)

# 拆单算法
exec_algo = Exec()
twap_orders = exec_algo.twap(total_qty, start_time, end_time)
```

**核心组件:**
- OrderBook: L2 订单簿重建 (支持 10 档)
- IcebergDetector: CUSUM 异常检测，识别隐藏大单
- SniperEngine: 多因素复合信号 (Spread + Imbalance + Volume)
- 算法交易: TWAP, VWAP, Sniper, Iceberg

### EventBacktest (hyperion/engine/ultra_backtest.py)

事件驱动回测引擎，全面超越 QLib 简单循环。

```python
from hyperion.engine.ultra_backtest import Cerebro, Strategy

# 事件驱动回测
cerebro = Cerebro(cash=1_000_000, commission=0.0003, slippage=0.001)
cerebro.add_strategy(MyStrategy())
cerebro.add_data(df)
results = cerebro.run()
```

**超越 QLib 的点:**
- 撮合引擎: 限价/市价/止损单
- 滑点模型: 线性 + 非线性冲击 (Kyle/Almgren)
- 成本模型: 佣金 + 冲击 + 资金成本
- Monte Carlo: 随机重排稳健性仿真

### OnlineLearning (hyperion/online/ultra_online.py)

在线学习 + 漂移检测。QLib 无此模块。

```python
from hyperion.online.ultra_online import OnlineLearningPipeline

pipeline = OnlineLearningPipeline(models)
pipeline.update(date, predictions, true_values, features)
```

**核心能力:**
- KS 漂移检测: Kolmogorov-Smirnov 分布检验
- KL Divergence: 信息论距离检测
- 滚动统计: 均值/方差/偏度/峰度 Z-score 监控
- 市场状态检测: Bull/Bear/HighVol/MediumVol/Trend
- 自适应重训练: IC 衰减自动触发

### PortfolioOptimizer (hyperion/portfolio/ultra_optimizer.py)

4+ 组合优化器。

```python
from hyperion.portfolio.ultra_optimizer import OptimizerFactory

# 创建优化器
opt = OptimizerFactory.create("hrp")
weights = opt.optimize(expected_returns, cov_matrix)
```

**优化器:**
- Risk Budgeting: 自定义风险预算分配
- HRP: Hierarchical Risk Parity 层次聚类
- Mean-CVaR: 最小化尾部风险
- OnlineAdaptive: 根据市场状态自适应

## 开发环境

### 系统要求

- Python: 3.10+
- OS: Linux/macOS/Windows WSL2
- CPU: 4 核心以上
- RAM: 8GB+

### 安装

```bash
git clone <repo_url>
cd quant_trading
pip install -r requirements.txt
```

### 验证安装

```bash
python -m tests.v27_ultra_benchmark
```

## 构建与测试

### 运行全部测试

```bash
# 全部测试
python -m tests.v27_ultra_benchmark

# 对比验证
python -m tests.test_qlib_equivalence
```

### 检查依赖

```python
python -c "from hyperion import check_dependencies; print(check_dependencies())"
```

## API参考

### 快速入门

```python
import pandas as pd
from hyperion import HyperionEngine

# 配置
config = HyperionConfig(symbols=["000300.SH"], model_names=["lgb"])

# 初始化并从新编排
engine = HyperionEngine(config)

# 训练
engine.train(data)

# 回测
results = engine.backtest(data)
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
```

## 贡献指南

### 编码规范

- PEP8
- Type hints 建议
- Docstrings: Google Style

### 提交前必做

1. 运行 `python -m tests.v27_ultra_benchmark`
2. 确保全部通过
3. 更新 `README.md`

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v25 | 2025-04 | Alpha Hunter v25 |
| v26 | 2025-05 | Hyperion Engine 初版 |
| v27 | 2025-05 | 全面超越 QLib |
