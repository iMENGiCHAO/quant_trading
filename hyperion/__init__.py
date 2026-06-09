"""
Hyperion Pro — 实战量化交易系统
=================================
A股实战量化交易框架，聚焦实用性与投资指导价值。

核心模块:
  data          → 行情数据 (Sina API 实时 + 历史) + 基本面分析
  strategy      → 5大策略引擎 (趋势/均值回归/动量/成交量/多因子)
  engine        → 事件驱动回测 (Sharpe/Sortino/Calmar)
  analysis      → 投资决策引擎 + 技术分析 + 市场状态 + 信号生成
  risk          → 风险管理 (VaR/头寸控制/止损)
  dashboard     → Dash 可视化仪表板
  reporting     → 自动化报告生成

使用:
    $ python hyperion/cli.py                  # 完整市场分析
    $ python hyperion/cli.py --stock 600519   # 个股深度分析
    $ python hyperion/cli.py --decision       # 投资决策报告
    $ python hyperion/dashboard/app.py        # Web 仪表板
"""

__version__ = "2.0.0"
__author__ = "Hyperion Quant Team"
__license__ = "MIT"

__all__ = [
    "version",
    "check_dependencies",
    "get_system_info",
]


def version() -> str:
    """返回版本号"""
    return __version__


def check_dependencies() -> dict:
    """检查所有依赖是否安装"""
    deps = {
        "numpy": False,
        "pandas": False,
        "scipy": False,
        "sklearn": False,
        "lightgbm": False,
        "xgboost": False,
        "akshare": False,
        "dash": False,
        "plotly": False,
    }
    for name in deps:
        try:
            __import__(name if name != "sklearn" else "sklearn")
            deps[name] = True
        except ImportError:
            pass
    return deps


def get_system_info() -> str:
    """返回系统信息"""
    deps = check_dependencies()
    all_ok = all(deps.values())

    info = f"""
╔══════════════════════════════════════╗
║  Hyperion Pro  v{__version__:<21} ║
╠══════════════════════════════════════╣
║  Status: {'Fully Ready' if all_ok else 'Partial'}            ║
╠══════════════════════════════════════╣
║  Core Modules:                       ║
║  • Data Layer    Sina API 实时+历史  ║
║  • Strategy      5大交易策略         ║
║  • Backtest      事件驱动回测引擎    ║
║  • Decision      投资决策引擎        ║
║  • Risk Manager  VaR + 头寸控制      ║
║  • Dashboard     Dash Web仪表板      ║
║  • Reporting     自动化报告生成      ║
╠══════════════════════════════════════╣
║  Dependencies:                       ║
║  numpy    {'✅' if deps.get('numpy') else '❌'}                     ║
║  pandas   {'✅' if deps.get('pandas') else '❌'}                     ║
║  scipy    {'✅' if deps.get('scipy') else '❌'}                     ║
║  sklearn  {'✅' if deps.get('sklearn') else '❌'}                     ║
║  lightgbm {'✅' if deps.get('lightgbm') else '❌'}                     ║
║  xgboost  {'✅' if deps.get('xgboost') else '❌'}                     ║
║  akshare  {'✅' if deps.get('akshare') else '❌'}                     ║
║  dash     {'✅' if deps.get('dash') else '❌'}                     ║
║  plotly   {'✅' if deps.get('plotly') else '❌'}                     ║
╠══════════════════════════════════════╣
║  Quick Start:                        ║
║  python hyperion/cli.py              ║
╚══════════════════════════════════════╝
"""
    return info
