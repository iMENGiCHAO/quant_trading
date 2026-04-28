"""
Hyperion Analysis Layer (Layer 7)
===================================
融合:
  Backtrader Analyzer → 回测分析
  VectorBT → 组合统计
  Qlib → IC/IR分析
"""
from hyperion.analysis.metrics import PerformanceMetrics
from hyperion.analysis.report import ReportGenerator

__all__ = ["PerformanceMetrics", "ReportGenerator"]
