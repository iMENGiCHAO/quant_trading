"""
Hyperion+ v27 — Production-Grade Quantitative Trading Framework
================================================================
全面超越 Microsoft QLib 的量化交易系统。

核心模块:
  alpha         → AlphaUltra 超级因子引擎 (698+ 因子)
  model_zoo     → ModelZoo Ultra (14+ 模型 + 前沿)
  hft           → 高频/微观结构引擎 (QLib 无)
  engine        → 事件驱动回测 + 撮合
  online        → 在线学习 + 漂移检测 (QLib 无)
  portfolio     → 组合优化 (4+ 优化器)
  orchestrator  → 全链路编排器

使用:
    >>> from hyperion import HyperionEngine
    >>> config = OrchestratorConfig()
    >>> engine = HyperionEngine(config)
"""

__version__ = "27.0.0"
__author__ = "Hyperion Quant Team"
__license__ = "Proprietary"

# 显式导出
def _get_orchestrator():
    """懒加载"""
    try:
        from .ultra_orchestrator import UltraOrchestrator, OrchestratorConfig
        return UltraOrchestrator, OrchestratorConfig
    except ImportError:
        return None, None

# 尝试暴露核心类
_lazy_loaded = False

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
║  Hyperion+ v{__version__:<30} ║
╠══════════════════════════════════════╣
║  Status: {'Ultra Ready' if all_ok else 'Partial'}       ║
╠══════════════════════════════════════╣
║  Core Modules:                       ║
║  • AlphaUltra      698+ factors      ║
║  • ModelZoo Ultra  14+ models        ║
║  • HF Engine       Tick-level        ║
║  • Event Backtest  Monte Carlo       ║
║  • Online Learning Drift detection   ║
║  • Portfolio Opt   4+ optimizers     ║
║  • Orchestrator    Full pipeline     ║
╠══════════════════════════════════════╣
║  Dependencies:                       ║
║  numpy    {'✅' if deps.get('numpy') else '❌'}                      ║
║  pandas   {'✅' if deps.get('pandas') else '❌'}                      ║
║  scipy    {'✅' if deps.get('scipy') else '❌'}                      ║
║  sklearn  {'✅' if deps.get('sklearn') else '❌'}                      ║
║  lightgbm {'✅' if deps.get('lightgbm') else '❌'}                      ║
║  xgboost  {'✅' if deps.get('xgboost') else '❌'}                      ║
╠══════════════════════════════════════╣
║  Quick Start:                        ║
║  python -m tests.v27_ultra_benchmark║
╚══════════════════════════════════════╝
"""
    return info
