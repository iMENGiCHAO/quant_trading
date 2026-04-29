# Hyperion Quant v1.0

> **融合 GitHub Top 12 量化交易项目精华，打造专业 A 股量化交易研究框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 设计理念

Hyperion Quant 不是"又一个量化框架"，而是**系统性融合**了 GitHub 上最热门的 12 个量化交易项目的精华：

| 来源 | Stars | 贡献 |
|------|-------|------|
| **Qlib** (Microsoft) | 16k★ | Alpha158因子、数据层、qrun工作流 |
| **VnPy/VeighNa** | 29k★ | 事件驱动引擎、CTA策略、风控 |
| **Freqtrade** | 32k★ | CLI工具、Hyperopt超参优化 |
| **FinRL** | 14k★ | RL交易环境、DRL Agent |
| **VectorBT** | 5k★ | 向量化回测、组合分析 |
| **Backtrader** | 15k★ | Cerebro引擎、Analyzer模式 |
| **Jesse** | 6k★ | 极简策略API、Monte Carlo |
| **Zipline-Reloaded** | 3.5k★ | Pipeline因子研究 |
| **Hummingbot** | 8k★ | Strategy-Connector分离 |
| **QuantConnect/LEAN** | 10k★ | 企业级架构设计 |

## 🏗️ 架构 (7层设计)

```
Layer 1: DATA      Qlib DataServer + Freqtrade数据管理
Layer 2: ALPHA     Alpha158因子 + 贝叶斯权重 + 因果发现
Layer 3: STRATEGY  CTA模板 + ML多因子 + RL环境
Layer 4: ENGINE    事件驱动 + 回测引擎 + 超参优化
Layer 5: RISK      风控 + 组合优化 (HRP/Risk Budgeting)
Layer 6: EXECUTION Broker抽象 + 模拟交易
Layer 7: ANALYSIS  性能指标 + 报告生成
```

## 🚀 快速开始

```bash
# 安装
pip install -r requirements.txt

# 下载数据
python -m hyperion.cli download --symbols 000001.SZ,000002.SZ --start 2024-01-01 --end 2024-12-31

# 运行回测
python -m hyperion.cli backtest

# 超参优化
python -m hyperion.cli hyperopt --evals 200
```

## 📊 核心特性

- **Alpha158 因子库**: 158个经过验证的量价因子
- **事件驱动引擎**: VnPy风格的高性能事件总线
- **回测/实盘统一**: 同一套策略代码兼容回测和实盘
- **A股约束**: T+1、涨跌停、ST过滤、印花税
- **贝叶斯在线学习**: 动态调整因子权重
- **因果发现**: Granger因果检验筛选有效因子
- **多种组合优化**: HRP / Risk Budgeting / Mean-Variance / Max Sharpe
- **RL交易环境**: FinRL风格Gym环境
- **超参优化**: Optuna Bayesian优化 + TimeSeriesSplit CV
- **CLI工具**: Freqtrade风格的命令行界面

## 📁 项目结构

```
quant_trading/
├── hyperion/            # 核心框架
│   ├── config.py        # Pydantic配置系统
│   ├── cli.py           # CLI入口
│   ├── data/            # Layer 1: 数据层
│   │   ├── server.py        # 高性能DataServer
│   │   ├── cache.py         # Parquet缓存
│   │   └── sources/         # 数据源适配器
│   ├── alpha/           # Layer 2: 因子层
│   │   ├── factors.py       # Alpha158因子
│   │   ├── technical.py     # 技术指标工厂
│   │   ├── bayesian.py      # 贝叶斯学习
│   │   └── causal.py        # 因果发现
│   ├── strategy/        # Layer 3: 策略层
│   │   ├── base.py          # 策略基类
│   │   ├── ml_strategy.py   # ML多因子选股
│   │   └── rl/env.py        # RL交易环境
│   ├── engine/          # Layer 4: 引擎层
│   │   ├── event_engine.py  # 事件驱动引擎
│   │   ├── backtest.py      # 回测引擎
│   │   └── hyperopt.py      # 超参优化
│   ├── risk/            # Layer 5: 风控层
│   │   ├── manager.py       # 风控管理
│   │   └── optimizer.py     # 组合优化
│   ├── execution/       # Layer 6: 执行层
│   │   ├── broker.py        # Broker抽象
│   │   └── simulator.py     # 模拟交易
│   └── analysis/        # Layer 7: 分析层
│       ├── metrics.py       # 性能指标
│       └── report.py        # 报告生成
├── config.yaml          # 默认配置
├── requirements.txt
└── README.md
```

## ⚠️ 免责声明

本软件仅供教育和研究目的。**使用风险自负。** 量化交易涉及重大风险，历史回测不代表未来收益。

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)
