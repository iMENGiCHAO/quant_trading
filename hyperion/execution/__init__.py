"""
Hyperion Execution Layer (Layer 6)
====================================
融合:
  Hummingbot → Strategy-Connector分离
  VnPy → 多网关适配
"""
from hyperion.execution.broker import Broker
from hyperion.execution.simulator import PaperBroker

__all__ = ["Broker", "PaperBroker"]
