"""Hyperion Pro 数据层"""
from __future__ import annotations

import os
import json
import socket
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Callable
import pandas as pd
import numpy as np

# 数据目录
DATA_DIR = Path.home() / ".hyperion_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# akShare 可用性检测
_HAS_AKSHARE = False
try:
    import akshare as ak
    _HAS_AKSHARE = True
except ImportError:
    pass

# A股核心股票池 (按行业分类，覆盖各板块龙头)
CORE_STOCKS = [
    ("600519", "贵州茅台", "食品饮料"),
    ("000858", "五粮液", "食品饮料"),
    ("000568", "泸州老窖", "食品饮料"),
    ("600809", "山西汾酒", "食品饮料"),
    ("600036", "招商银行", "银行"),
    ("601398", "工商银行", "银行"),
    ("601939", "建设银行", "银行"),
    ("601166", "兴业银行", "银行"),
    ("601318", "中国平安", "保险"),
    ("601628", "中国人寿", "保险"),
    ("600030", "中信证券", "证券"),
    ("601688", "华泰证券", "证券"),
    ("300059", "东方财富", "互联网金融"),
    ("000333", "美的集团", "家电"),
    ("000651", "格力电器", "家电"),
    ("600690", "海尔智家", "家电"),
    ("300750", "宁德时代", "新能源"),
    ("601012", "隆基绿能", "新能源"),
    ("600900", "长江电力", "电力"),
    ("600585", "海螺水泥", "建材"),
    ("002415", "海康威视", "安防"),
    ("000002", "万科A", "房地产"),
    ("600048", "保利发展", "房地产"),
    ("600276", "恒瑞医药", "医药"),
    ("603259", "药明康德", "医药"),
    ("002475", "立讯精密", "消费电子"),
    ("000725", "京东方A", "面板"),
    ("002714", "牧原股份", "养猪"),
    ("601899", "紫金矿业", "有色"),
    ("600887", "伊利股份", "乳业"),
    ("000001", "平安银行", "银行"),
    ("002352", "顺丰控股", "物流"),
    ("688981", "中芯国际", "半导体"),
    ("603501", "韦尔股份", "半导体"),
    ("002594", "比亚迪", "新能源汽车"),
    ("601985", "中国核电", "电力"),
    ("601088", "中国神华", "煤炭"),
    ("600028", "中国石化", "石油"),
    ("601857", "中国石油", "石油"),
    ("601728", "中国电信", "通信"),
    ("600941", "中国移动", "通信"),
    ("002230", "科大讯飞", "人工智能"),
    ("300760", "迈瑞医疗", "医疗器械"),
    ("300015", "爱尔眼科", "医疗服务"),
    ("603288", "海天味业", "调味品"),
    ("000538", "云南白药", "中药"),
    ("600196", "复星医药", "医药"),
    ("300124", "汇川技术", "工业自动化"),
    ("002129", "中环股份", "光伏"),
]

# 指数列表
INDICES = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000016.SH": "上证50",
    "399673.SZ": "创业板50",
}


def _check_network() -> bool:
    """检查网络连通性"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


def has_akshare() -> bool:
    """检查akShare是否可用（安装+网络）"""
    return _HAS_AKSHARE and _check_network()


def get_stock_name(code: str) -> str:
    """根据代码获取股票名称"""
    for c, name, *_ in CORE_STOCKS:
        if c == code:
            return name
    return code


def get_stock_industry(code: str) -> str:
    """获取股票行业"""
    for c, _, ind in CORE_STOCKS:
        if c == code:
            return ind
    return "未知"


def get_stocks_by_industry(industry: str) -> List[str]:
    """按行业获取股票列表"""
    return [c for c, _, ind in CORE_STOCKS if ind == industry]


def list_industries() -> List[str]:
    """获取所有行业列表"""
    return sorted(set(ind for _, _, ind in CORE_STOCKS))


def industry_summary() -> Dict[str, List[str]]:
    """行业-股票映射"""
    result = {}
    for c, name, ind in CORE_STOCKS:
        result.setdefault(ind, []).append((c, name))
    return result
