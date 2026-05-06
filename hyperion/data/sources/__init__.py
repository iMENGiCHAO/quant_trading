"""
数据源适配器 (VnPy多数据源 + Freqtrade exchange abstraction)
=========================================================
支持: AkShare(免费), TuShare, BaoStock, CSV, SQLite本地
"""
from hyperion.data.sources.base import BaseDataSource
from hyperion.data.sources.akshare import AkShareSource

_SOURCE_REGISTRY = {
    "akshare": AkShareSource,
}

def get_source(name: str, **kwargs) -> BaseDataSource:
    """获取数据源实例"""
    if name not in _SOURCE_REGISTRY:
        available = list(_SOURCE_REGISTRY.keys())
        raise ValueError(f"Unknown source '{name}'. Available: {available}")
    return _SOURCE_REGISTRY[name](**kwargs)

def register_source(name: str, cls):
    """注册自定义数据源"""
    _SOURCE_REGISTRY[name] = cls

__all__ = ["BaseDataSource", "AkShareSource", "get_source", "register_source"]
