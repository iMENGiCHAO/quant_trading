# Hyperion Pro — A股实战量化交易系统

> **从"数据可视化"到"可执行投资决策"——这不是又一个画K线的工具，这是你的AI交易副手。**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data Source](https://img.shields.io/badge/Data-Sina%20Finance%20%2B%20akShare-orange)](https://finance.sina.com.cn)

---

## 目录

- [为什么这个系统不一样？](#为什么这个系统不一样)
- [快速开始](#快速开始)
- [核心能力矩阵](#核心能力矩阵)
- [命令行使用指南](#命令行使用指南)
- [Web 仪表盘](#web-仪表盘)
- [系统架构](#系统架构)
- [实战案例：如何用这个系统赚钱](#实战案例如何用这个系统赚钱)
- [性能与数据真实性声明](#性能与数据真实性声明)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 为什么这个系统不一样？

市面上大多数"量化交易系统"止步于数据可视化——画几条均线、展示K线图，然后让用户自己判断。Hyperion Pro 的核心差异在于：

```
典型量化系统:   "MACD金叉了" → 用户自己决定怎么办
Hyperion Pro:  "买入 600519 (贵州茅台), 现价 ¥1850,
                目标 ¥2100, 止损 ¥1750, 仓位 15%,
                盈亏比 2.5:1, 入场区间 ¥1795-¥1820,
                若放量突破前高加仓至 20%"
```

**每一行输出都是一条可执行的交易指令**，而不是一句模棱两可的分析结论。

### 核心设计理念

| 维度 | 传统量化系统 | Hyperion Pro |
|------|------------|-------------|
| 分析输出 | MACD金叉、KDJ超卖 | **买入/卖出/持有** + 具体价格区间 |
| 仓位建议 | 无 | **凯利公式优化**仓位比例 |
| 风险控制 | 无 | 硬止损 + 移动止损 + 阶梯止盈 |
| 绩效追踪 | 无 | **信号命中率**、策略归因、行为分析 |
| 数据源 | 可能混用模拟数据 | **纯真实数据**，验证失败明确报错 |

---

## 快速开始

### 环境要求

- Python 3.10+（推荐 3.12）
- 网络连接（访问新浪财经 API）

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/iMENGiCHAO/quant_trading.git
cd quant_trading

# 2. 安装依赖（推荐使用虚拟环境）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖包
pip install -r requirements.txt

# 4. 运行完整投资简报
python hyperion/cli.py
```

### 依赖清单

```
numpy>=1.24.0        # 数值计算
pandas>=2.0.0        # 数据处理
scipy>=1.10.0        # 科学计算
akshare>=1.14.0      # A股数据源
dash>=2.14.0         # Web仪表盘
plotly>=5.18.0       # 交互式图表
statsmodels>=0.14.0  # 统计模型
```

---

## 核心能力矩阵

### 📊 数据层 — 双数据源，绝不伪造

数据是量化交易的基石。Hyperion Pro 从两个真实数据源获取信息，**绝不静默降级、绝不返回假数据**。

- **新浪财经 API**（实时）：实时行情、历史K线、板块数据
- **akShare 开源库**（备选）：基本面、资金流向
- **数据新鲜度标记**：每个决策都标注数据时间和来源

### 🧠 分析层 — 五大策略引擎

| 策略 | 类型 | 适用场景 | 核心指标 |
|------|------|---------|---------|
| TrendFollowing | 趋势跟踪 | 单边行情 | MA排列 / MACD / ADX |
| MeanReversion | 均值回归 | 震荡行情 | RSI / 布林带 / 乖离率 |
| MomentumBreakout | 动量突破 | 强势市场 | 价格通道 / 成交量 |
| VolumeAnomaly | 成交量异动 | 短线交易 | 量比 / 资金流 |
| MultiFactorAlpha | 多因子 | 中长线配置 | 技术+基本面+资金面综合 |

### 📈 决策层 — 完整的可执行投资指令

每个 `InvestmentDecision` 输出包含：

```
📋 操作计划

1️⃣ 入场策略: ¥1795 - ¥1820 区间分批建仓
   - 第一笔(40%仓位)在 ¥1820 附近
   - 第二笔(30%仓位)若回踩 ¥1795 加仓
   - 剩余30%仓位等待放量突破确认后追加

2️⃣ 仓位管理: 总仓位不超过总资金的 15%

3️⃣ 止损纪律: ¥1750 (硬止损，触及立即离场)
   - 移动止损: 浮盈超 5% 后上移至成本
   - 时间止损: 中线持有，30天后重新评估

4️⃣ 阶梯止盈:
   - 第一目标 ¥1950 → 平仓 50%
   - 第二目标 ¥2100 → 清仓离场

5️⃣ 关键观察: 量比 > 1.5、MACD柱状线放大、板块联动
```

### 🚨 预警层 — 实时信号扫描

`SignalAlertSystem` 实时监测五大维度，每个预警附带**建议操作**：

| 预警类型 | 示例 | 操作建议 |
|---------|------|---------|
| 情绪过热 | 上涨比例>80% | 减仓获利，不追高 |
| MACD底背离 | 价格新低但动能未新低 | 强烈建议关注！逢低建仓 |
| 板块轮动 | 板块严重分化 | 跟随最强板块或等待轮动 |
| 放量突破 | 量比>3+涨幅>3% | 放量突破有效，可追涨 |
| 指数破位 | 跌破60日均线 | 大幅减仓至30%以下 |

### 📝 交易日志层 — 记录每一笔交易

`TradeJournal` 系统帮助用户追踪自己的交易行为：

- **归因分析**：赚钱是因为系统信号还是运气？
- **行为分析**：是否追涨杀跌？是否过早止盈？
- **月度报告**：胜率、盈亏比、情绪状态
- **系统信任度**：跟随系统 vs 自主决策的胜率对比

### 🎯 绩效追踪层 — 系统自己的"成绩单"

`PerformanceTracker` 自动记录所有系统发出的投资建议，并在T+5天后回溯验证：

- **信号命中率**：系统说"买入"时，真的涨了没有？
- **策略归因**：哪个策略最赚钱？哪个在亏钱？
- **行业归因**：在哪个行业上预测最准？
- **累计绩效**：所有信号的整体期望值

---

## 命令行使用指南

```bash
# 完整投资简报（市场状态 + Top 10 + 风险预警）
python hyperion/cli.py

# 个股深度分析（含完整操作计划）
python hyperion/cli.py --stock 600519
python hyperion/cli.py --stock 600519 000858 300750  # 多只同时分析

# 最佳投资标的（可指定数量）
python hyperion/cli.py --top 20

# 风险预警
python hyperion/cli.py --risk

# 仅查看市场状态
python hyperion/cli.py --market

# 策略回测
python hyperion/cli.py --backtest  # 默认 茅台 五粮液 宁德
python hyperion/cli.py --backtest 600519 000858 --bt-days 120

# 🚨 实时预警扫描（新增）
python hyperion/cli.py --alerts
python hyperion/cli.py --alerts --alert-level CRITICAL  # 只看紧急

# 📝 交易日志（新增）
python hyperion/cli.py --journal                                     # 绩效概览
python hyperion/cli.py --journal-entry 600519 1850 100 "MACD金叉"    # 记录买入
python hyperion/cli.py --journal-exit trade_id 2100 "目标达成止盈"   # 记录卖出

# 📊 月度报告（新增）
python hyperion/cli.py --monthly 2026-06
```

---

## Web 仪表盘

启动交互式 Web 仪表盘：

```bash
python hyperion/dashboard/app.py
# 打开浏览器 → http://127.0.0.1:8050
```

### 面板一览

| 标签页 | 功能 |
|-------|------|
| 市场总览 | 市场状态、指数行情、技术指标、量能分析、操作指南 |
| 投资决策 | 买入信号卡片（含评分、目标价、止损、仓位）、风险预警 |
| 策略回测 | 多策略对比排名（夏普/胜率/最大回撤） |
| 绩效追踪 | 信号命中率、策略绩效排名、最近验证信号 |
| 风险管理 | VaR/压力测试/分散化/风险警告 |
| 报告中心 | 已生成报告列表 |
| **信号预警** 🆕 | 紧急预警/预警信号/市场提示，每条含操作建议 |
| **交易日志** 🆕 | 持仓管理、历史平仓、系统信号跟踪分析 |

---

## 系统架构

```
quant_trading/
├── hyperion/
│   ├── cli.py                          # 🚀 命令行入口
│   ├── data/
│   │   ├── market.py                   # 📊 数据获取（Sina API + akShare）
│   │   └── fundamental.py              #  基本面数据
│   ├── analysis/
│   │   ├── technical.py                # 📉 技术指标（MA/MACD/RSI/KDJ/布林带）
│   │   ├── market_state.py             # 🌍 市场状态判断
│   │   ├── decision_engine.py          # ⚡ 投资决策引擎（核心）
│   │   ├── signals.py                  #  信号生成器
│   │   ├── signal_alerts.py            # 🚨 实时预警系统 **
│   │   └── trade_journal.py            # 📝 交易日志系统 **
│   ├── strategy/
│   │   ├── base.py                     #  策略基类
│   │   └── strategies.py               #   5大策略实现
│   ├── engine/
│   │   └── backtest.py                 # 🔙 回测引擎
│   ├── risk/
│   │   └── manager.py                  # 🛡 风险管理（VaR/压力测试）
│   ├── performance/
│   │   └── tracker.py                  # 📈 信号绩效追踪
│   ├── reporting/
│   │   └── report_generator.py         # 📄 报告生成
│   └── dashboard/
│       └── app.py                      # 🖥 Web仪表盘（8个Tab）
├── requirements.txt
└── README.md
```

> ** 新增模块

---

## 实战案例：如何用这个系统赚钱

### 场景一：每日复盘（5分钟）

```bash
# 早上9:25 开盘前
python hyperion/cli.py              # 查看市场状态 + 最佳标的 + 风险
python hyperion/cli.py --alerts     # 扫描实时预警
```

系统会告诉你：
- 今天市场处于什么状态（牛市/震荡/熊市）
- 推荐仓位比例（满仓/半仓/空仓）
- 哪些板块最强、哪些需要回避
- **具体哪只股票值得买入**，买多少、什么价止损

### 场景二：个股分析（立即决策）

```bash
python hyperion/cli.py --stock 600519
```

输出包含完整操作计划：入场价格区间、仓位、止损、止盈、持有期限。**对照执行即可。**

### 场景三：绩效复盘（每月一次）

```bash
python hyperion/cli.py --journal       # 查看总体绩效
python hyperion/cli.py --monthly 2026-06  # 月度报告
```

系统会对比"跟随系统信号"和"自主决策"的胜率差异。如果跟随系统胜率更高——说明你应该更信任系统。

### 场景四：盘中预警（实时盯盘）

```bash
# 随时运行
python hyperion/cli.py --alerts --alert-level WARNING
```

系统扫描五大维度预警：
- 发现 MACD 底背离 → **强烈建议关注，逢低建仓**
- 发现指数跌破 MA60 → **大幅减仓至30%以下**
- 发现板块轮动 → **跟随最强板块**

---

## 性能与数据真实性声明

### ✅ 数据真实性

- **所有数据来自真实 API**（新浪财经、akShare）
- **绝不静默降级**：当API不可用时，系统明确报错，不返回假数据
- **每条决策标注数据新鲜度**：数据来源 + 更新时间

### ⚠️ 风险提示

量化交易系统是投资工具，**不是稳赚不赔的印钞机**。Hyperion Pro 的设计目标是：

1. **提高胜率**：通过多因子评分 + 策略共识，筛选出大概率上涨的标的
2. **控制风险**：严格止损 + 凯利公式仓位优化，单笔风险可控
3. **持续改进**：绩效追踪系统让用户知道自己系统的真实表现

**过往表现不代表未来收益。** 任何声称"稳赚"的量化系统都是骗局。

---

## 常见问题

**Q: 需要每天手动运行吗？**
A: 建议开盘前运行一次 `python hyperion/cli.py` 查看当日策略，盘中可运行 `--alerts` 扫描预警。

**Q: 数据延迟多少？**
A: 新浪财经 API 提供**实时数据**，延迟通常在 3-5 秒以内。

**Q: 有回测功能吗？**
A: 有。`--backtest` 参数支持对比5大策略的夏普比率、胜率、最大回撤等指标。

**Q: 这个系统真的能赚钱吗？**
A: Hyperion Pro 提供的是**高质量的决策支持**，最终执行取决于用户。系统的绩效追踪模块会持续统计信号命中率——让数据说话。

---

## 许可证

MIT License — 自由使用、修改和分发。

如果这个项目对你有所帮助，欢迎 ⭐️ Star 支持。
