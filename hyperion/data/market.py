
"""
Hyperion Pro — 市场数据引擎
=============================
多源设计：
  - 主数据源：Sina Finance API（实时行情 + K线）
  - 备用数据源：akShare（估值数据、板块数据）
  - 缓存层：6小时K线缓存，1小时行情缓存

核心输出：
  - 实时行情快照
  - 历史K线数据
  - 指数行情
  - 行业板块行情
"""
from __future__ import annotations

import json
import os
import time
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import requests

from . import (
    CORE_STOCKS, INDICES, DATA_DIR, CACHE_DIR,
    get_stock_name, get_stock_industry,
)

# ==========================================================
#  缓存管理
# ==========================================================
MARKET_CACHE_TTL = 3600       # 行情缓存1小时
HISTORY_CACHE_TTL = 21600     # K线缓存6小时


def _cache_key(prefix: str, *args) -> Path:
    key = "_".join(str(a).replace("/", "_").replace(".", "_") for a in args)
    return CACHE_DIR / f"{prefix}_{key}.pkl"


def _load_cache(path: Path, max_age: float = None) -> Optional[Any]:
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if max_age and age > max_age:
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError):
        return None


def _save_cache(path: Path, data: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


# ==========================================================
#  Sina Finance API — Primary data source
# ==========================================================
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_KLINE_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
)
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn"}

# Sina symbol mapping
_SINA_PREFIX = {}
for _c, _n, _i in CORE_STOCKS:
    _SINA_PREFIX[_c] = "sh" + _c if _c.startswith(("6", "9")) else "sz" + _c
for _k in INDICES:
    _SINA_PREFIX[_k] = _k.replace(".SH", ".SH").replace(".SZ", ".SZ")


def _to_sina_symbol(code: str) -> str:
    """Convert internal code to Sina symbol"""
    if code in _SINA_PREFIX:
        return _SINA_PREFIX[code]
    return ("sh" + code) if code.startswith(("6", "9")) else ("sz" + code)


# ==========================================================
#  Real-time quote session (keep-alive)
# ==========================================================
_QUOTE_SESSION = None

def _get_session():
    global _QUOTE_SESSION
    if _QUOTE_SESSION is None:
        _QUOTE_SESSION = requests.Session()
        _QUOTE_SESSION.headers.update(SINA_HEADERS)
        # Disable system proxy to avoid Eastmoney proxy errors
        _QUOTE_SESSION.trust_env = False
    return _QUOTE_SESSION


# ==========================================================
#  实时行情
# ==========================================================

def fetch_realtime_quotes(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    获取A股实时行情快照 — Sina主源
    
    Returns:
        DataFrame: [code, name, price, change_pct, volume, amount, ...]
    """
    cache_key = _cache_key("realtime", "sina")
    cached = _load_cache(cache_key, 60)  # 1分钟缓存即可
    if cached is not None:
        return cached

    if symbols is None:
        symbols = [c for c, _, _ in CORE_STOCKS]

    try:
        rows = []
        session = _get_session()
        BATCH_SIZE = 30
        
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            sina_codes = ",".join(_to_sina_symbol(c) for c in batch)
            url = SINA_QUOTE_URL.format(symbols=sina_codes)
            r = session.get(url, timeout=8)
            
            if r.status_code != 200:
                continue
                
            for line in r.text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    parts = line.split('"')
                    if len(parts) < 2:
                        continue
                    name_prefix = parts[0].split("_")[-1]
                    quote_str = parts[1]
                    if not quote_str:
                        continue
                    fields = quote_str.split(",")
                    if len(fields) < 32:
                        continue
                    
                    # Determine code from name_prefix
                    code = name_prefix[2:]  # strip sh/sz prefix
                    
                    rows.append({
                        "code": code,
                        "name": fields[0],
                        "open": float(fields[1]) if fields[1] else np.nan,
                        "pre_close": float(fields[2]) if fields[2] else np.nan,
                        "price": float(fields[3]) if fields[3] else np.nan,
                        "high": float(fields[4]) if fields[4] else np.nan,
                        "low": float(fields[5]) if fields[5] else np.nan,
                        "volume": int(float(fields[8])) if fields[8] else 0,
                        "amount": float(fields[9]) if fields[9] else 0.0,
                        "change_pct": round(
                            (float(fields[3]) / float(fields[2]) - 1) * 100, 2
                        ) if fields[3] and fields[2] else 0.0,
                        "change": round(
                            float(fields[3]) - float(fields[2]), 2
                        ) if fields[3] and fields[2] else 0.0,
                        "date": fields[30] if len(fields) > 30 else "",
                        "time": fields[31] if len(fields) > 31 else "",
                    })
                except (ValueError, IndexError):
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            # Add industry
            df["industry"] = df["code"].apply(
                lambda c: get_stock_industry(c) if len(get_stock_industry(c)) > 2 else "其他"
            )
            # Ensure name is correct
            df["name"] = df["code"].apply(get_stock_name)
            
            # Add basic fundamental fields (Sina doesn't provide these in real-time)
            df["pe_ttm"] = np.nan
            df["pb"] = np.nan
            df["total_mv"] = np.nan
            df["float_mv"] = np.nan
            df["turnover"] = np.nan
            
            _save_cache(cache_key, df)
            return df
            
    except Exception as e:
        print(f"[数据] Sina实时行情获取失败: {e}")
    
    # Fallback to akShare
    return _fetch_realtime_akshare(symbols)


def _fetch_realtime_akshare(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """akShare fallback for real-time quotes"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        
        col_map = {
            "代码": "code", "名称": "name",
            "最新价": "price", "涨跌幅": "change_pct",
            "涨跌额": "change", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover",
            "市盈率-动态": "pe_ttm", "市净率": "pb",
            "总市值": "total_mv", "流通市值": "float_mv",
            "今开": "open", "最高": "high", "最低": "low",
            "昨收": "pre_close",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        
        for col in ["price", "change_pct", "volume", "amount", "turnover",
                   "pe_ttm", "pb", "total_mv", "float_mv", "open", "high", "low", "pre_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        if symbols:
            df = df[df["code"].isin(symbols)].copy()
        
        df["industry"] = df["code"].apply(
            lambda c: get_stock_industry(c) if len(get_stock_industry(c)) > 2 else "其他"
        )
        df["name"] = df["code"].apply(get_stock_name)
        
        cache_key = _cache_key("realtime", "akshare")
        _save_cache(cache_key, df)
        return df
        
    except Exception as e:
        print(f"[数据] akShare实时行情获取失败: {e}")
        return _generate_demo_quotes(symbols)


def fetch_index_quotes() -> pd.DataFrame:
    """
    获取主要指数实时行情 — Sina主源
    """
    cache_key = _cache_key("index_quotes", "sina")
    cached = _load_cache(cache_key, 60)
    if cached is not None:
        return cached

    try:
        # Map our index codes to Sina format
        sina_index_map = {
            "000001.SH": "s_sh000001",
            "399001.SZ": "s_sz399001",
            "399006.SZ": "s_sz399006",
            "000688.SH": "s_sh000688",
            "000300.SH": "s_sh000300",
            "000905.SH": "s_sh000905",
            "000016.SH": "s_sh000016",
            "399673.SZ": "s_sz399673",
        }
        
        codes_str = ",".join(sina_index_map.values())
        session = _get_session()
        r = session.get(SINA_QUOTE_URL.format(symbols=codes_str), timeout=8)
        
        rows = []
        reverse_map = {v: k for k, v in sina_index_map.items()}
        
        for line in r.text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                parts = line.split('"')
                if len(parts) < 2:
                    continue
                name_prefix = parts[0].split("_")[-1]
                fields = parts[1].split(",")
                if len(fields) < 5:
                    continue
                
                sina_key = "s_" + name_prefix
                orig_code = reverse_map.get(sina_key, name_prefix)
                
                # Sina short-format (s_ prefix): name, price, change_amt, change_pct, volume, amount
                # fields[0]=name, [1]=current price, [2]=change_amount, [3]=change_pct, [4]=volume, [5]=amount
                current_price = float(fields[1]) if len(fields) > 1 and fields[1] else 0
                change_amount = float(fields[2]) if len(fields) > 2 and fields[2] else 0
                change_pct = float(fields[3]) if len(fields) > 3 and fields[3] else 0
                pre_close = current_price - change_amount
                if pre_close <= 0:
                    pre_close = current_price / (1 + change_pct / 100) if change_pct != -100 else current_price * 2
                calc_pct = round((current_price / pre_close - 1) * 100, 2) if pre_close > 0 else 0
                
                rows.append({
                    "code": orig_code,
                    "name": INDICES.get(orig_code, fields[0]),
                    "price": current_price,
                    "pre_close": pre_close,
                    "change_pct": calc_pct,
                    "change": round(current_price - pre_close, 2),
                    "volume": int(float(fields[4])) if len(fields) > 4 and fields[4] else 0,
                    "amount": float(fields[5]) if len(fields) > 5 and fields[5] else 0,
                })
            except (ValueError, IndexError):
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            _save_cache(cache_key, df)
            return df
            
    except Exception as e:
        print(f"[数据] Sina指数行情获取失败: {e}")
    
    return _generate_demo_index_quotes()


# ==========================================================
#  历史K线数据 — Sina主源
# ==========================================================

def fetch_history(symbol: str, days: int = 250) -> pd.DataFrame:
    """
    获取单只股票历史日K线 — Sina主源
    
    Args:
        symbol: 股票代码 (如 "600519")
        days: 回溯天数
        
    Returns:
        DataFrame: [date, open, high, low, close, volume, amount, change_pct]
    """
    cache_key = _cache_key("history", "sina", symbol, str(days))
    cached = _load_cache(cache_key, HISTORY_CACHE_TTL)
    if cached is not None and not cached.empty:
        return cached

    sina_sym = _to_sina_symbol(symbol)
    
    try:
        session = _get_session()
        url = SINA_KLINE_URL.format(symbol=sina_sym, datalen=days + 10)
        r = session.get(url, timeout=8)
        
        if r.status_code == 200 and r.text.strip():
            data = json.loads(r.text)
            if isinstance(data, list) and len(data) > 0:
                rows = []
                for bar in data:
                    try:
                        o = float(bar.get("open", 0))
                        h = float(bar.get("high", 0))
                        l = float(bar.get("low", 0))
                        c = float(bar.get("close", 0))
                        v = float(bar.get("volume", 0))
                        rows.append({
                            "date": pd.Timestamp(bar["day"]),
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": int(v),
                            "amount": c * v * 100 if v > 0 else 0.0,
                        })
                    except (ValueError, KeyError):
                        continue
                
                if rows:
                    df = pd.DataFrame(rows)
                    df.sort_values("date", inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    df["change_pct"] = df["close"].pct_change() * 100
                    df["change_pct"] = df["change_pct"].round(2)
                    df["amount"] = df["amount"].round(2)
                    
                    _save_cache(cache_key, df)
                    return df
        
    except Exception as e:
        print(f"[数据] Sina {symbol} K线获取失败: {e}")
    
    # Fallback to akShare
    return _fetch_history_akshare(symbol, days)


def _fetch_history_akshare(symbol: str, days: int = 250) -> pd.DataFrame:
    """akShare fallback for history"""
    try:
        import akshare as ak
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is not None and not df.empty:
            col_map = {
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "change_pct",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume", "amount", "change_pct"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
    except Exception as e:
        print(f"[数据] akShare {symbol} K线获取失败: {e}")
    
    return _generate_demo_history(symbol, days)


def fetch_batch_history(symbols: List[str], days: int = 250,
                        progress: bool = True) -> Dict[str, pd.DataFrame]:
    """批量获取历史数据（并行）"""
    result = {}
    total = len(symbols)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        fut_map = {executor.submit(fetch_history, s, days): s for s in symbols}
        for idx, fut in enumerate(as_completed(fut_map)):
            sym = fut_map[fut]
            if progress and idx % 10 == 0:
                print(f"  [{idx+1}/{total}] 加载 {sym}...", end="\r")
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    result[sym] = df
            except Exception as e:
                print(f"\n[数据] {sym} 获取失败: {e}")
    
    if progress:
        print(f"  [{total}/{total}] 完成{' '*20}")
    return result


# ==========================================================
#  Demo 模式 (离线可用)
# ==========================================================

def _generate_demo_quotes(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """生成仿真实时行情"""
    np.random.seed(int(time.time()) % 10000)
    
    rows = []
    stock_list = symbols if symbols else [c for c, _, _ in CORE_STOCKS]
    
    base_prices = {
        "600519": 1500, "000858": 130, "000568": 180, "600809": 220,
        "600036": 35, "601398": 6, "601939": 7, "601166": 18,
        "601318": 45, "601628": 35, "600030": 22, "601688": 15,
        "300059": 18, "000333": 65, "000651": 40, "600690": 28,
        "300750": 200, "601012": 25, "600900": 28, "600585": 25,
        "002415": 35, "000002": 10, "600048": 9, "600276": 45,
        "603259": 55, "002475": 30, "000725": 5, "002714": 45,
        "601899": 15, "600887": 28, "000001": 12, "002352": 40,
        "688981": 55, "603501": 100, "002594": 250, "601985": 10,
        "601088": 35, "600028": 6, "601857": 8, "601728": 5,
        "600941": 100, "002230": 60, "300760": 280, "300015": 15,
        "603288": 38, "000538": 50, "600196": 25, "300124": 65,
        "002129": 35,
    }
    
    for code in stock_list:
        if code not in base_prices:
            continue
        base = base_prices.get(code, 50)
        change_pct = np.random.normal(0, 1.5)
        price = base * (1 + change_pct / 100)
        
        rows.append({
            "code": code,
            "name": get_stock_name(code),
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "change": round(price - base, 2),
            "volume": int(np.random.lognormal(12, 1)),
            "amount": int(np.random.lognormal(17, 1)),
            "turnover": round(np.random.uniform(0.5, 5), 2),
            "pe_ttm": round(np.random.uniform(10, 50), 2),
            "pb": round(np.random.uniform(1, 8), 2),
            "total_mv": int(base * np.random.uniform(0.5, 2) * 1e8),
            "open": round(base * (1 + np.random.normal(0, 0.005)), 2),
            "high": round(base * (1 + abs(np.random.normal(0, 0.01))), 2),
            "low": round(base * (1 - abs(np.random.normal(0, 0.01))), 2),
            "pre_close": round(base, 2),
            "industry": get_stock_industry(code),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        _save_cache(_cache_key("realtime", "demo"), df)
    return df


def _generate_demo_index_quotes() -> pd.DataFrame:
    """生成仿真指数行情"""
    np.random.seed(int(time.time()) % 10000 + 100)
    
    base_values = {
        "000001.SH": 3300, "399001.SZ": 10500, "399006.SZ": 2100,
        "000688.SH": 950, "000300.SH": 3900, "000905.SH": 5800,
        "000016.SH": 2650, "399673.SZ": 1050,
    }
    
    rows = []
    for code, name in INDICES.items():
        base = base_values.get(code, 3000)
        change_pct = np.random.normal(0, 0.8)
        price = base * (1 + change_pct / 100)
        
        rows.append({
            "code": code, "name": name,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "change": round(price - base, 2),
            "volume": int(np.random.lognormal(15, 1)),
            "amount": int(np.random.lognormal(20, 1)),
        })
    
    return pd.DataFrame(rows)


def _generate_demo_history(symbol: str, days: int = 250) -> pd.DataFrame:
    """基于几何布朗运动生成仿真历史K线"""
    np.random.seed(hash(symbol) % (2**31))
    
    base_prices = {
        "600519": 1500, "000858": 130, "000568": 180, "600809": 220,
        "300750": 200, "601012": 25, "002594": 250, "688981": 55,
    }
    base = base_prices.get(symbol, 50)
    
    end_date = datetime.now()
    dates = pd.bdate_range(end=end_date, periods=min(days, 252*3))
    n = len(dates)
    
    industry = get_stock_industry(symbol)
    vol_map = {
        "银行": 0.015, "保险": 0.018, "证券": 0.022,
        "食品饮料": 0.018, "医药": 0.022, "新能源": 0.028,
        "半导体": 0.030, "家电": 0.016, "房地产": 0.025,
        "电力": 0.012, "煤炭": 0.018, "石油": 0.016,
    }
    mu_map = {
        "食品饮料": 0.0008, "医药": 0.0007, "新能源": 0.0005,
        "半导体": 0.0010, "银行": 0.0003, "保险": 0.0004,
    }
    
    sigma = vol_map.get(industry, 0.020)
    mu = mu_map.get(industry, 0.0004)
    
    dt = 1.0
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n)
    price_path = base * np.exp(np.cumsum(returns))
    price_path = np.maximum(price_path, base * 0.3)
    
    df = pd.DataFrame({"date": dates, "close": price_path})
    df["open"] = df["close"].shift(1) * (1 + np.random.normal(0, 0.003, n))
    df["open"] = df["open"].fillna(df["close"] * 0.99)
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
    df["volume"] = np.random.lognormal(12, 0.8, n).astype(int)
    df["amount"] = (df["volume"] * df["close"] * 100).astype(int)
    df["change_pct"] = df["close"].pct_change() * 100
    
    return df


# ==========================================================
#  板块/行业行情聚合
# ==========================================================

def sector_quotes() -> pd.DataFrame:
    """获取各行业板块汇总行情"""
    df = fetch_realtime_quotes()
    if df.empty or "industry" not in df.columns:
        return pd.DataFrame()
    
    grouped = df.groupby("industry").agg({
        "code": "count",
        "change_pct": "mean",
        "amount": "sum",
        "price": "mean",
    }).rename(columns={
        "code": "stock_count", "change_pct": "avg_change",
        "amount": "total_amount", "price": "avg_price",
    })
    
    up_count = df.groupby("industry").apply(
        lambda x: (x["change_pct"] > 0).sum()
    ).rename("up_stocks")
    down_count = df.groupby("industry").apply(
        lambda x: (x["change_pct"] < 0).sum()
    ).rename("down_stocks")
    
    grouped = grouped.join(up_count).join(down_count)
    grouped["up_ratio"] = grouped["up_stocks"] / (grouped["stock_count"] + 1e-12)
    grouped = grouped.sort_values("avg_change", ascending=False)
    
    return grouped


# ==========================================================
#  资金流向（基于量价关系估算）
# ==========================================================

def money_flow(symbol: str) -> dict:
    """
    基于量价关系估算资金流向
    
    真实资金流向需要Level-2数据，这里通过价格-成交量关系估算：
    - 放量上涨 → 主力流入
    - 放量下跌 → 主力流出
    - 缩量横盘 → 观望
    """
    try:
        df = fetch_history(symbol, days=30)
        if df.empty or len(df) < 5:
            return _demo_money_flow(symbol)
        
        close = df["close"].values
        volume = df["volume"].values
        
        price_changes = np.diff(close)
        vol_avg = np.mean(volume)
        
        recent_price_chg = price_changes[-5:]
        recent_vol = volume[-5:]
        
        # 量价关系分析
        up_days = recent_price_chg > 0
        down_days = recent_price_chg < 0
        
        up_vol = recent_vol[up_days].sum()
        down_vol = recent_vol[down_days].sum()
        total_vol = up_vol + down_vol
        
        if total_vol > 0:
            main_force_ratio = (up_vol - down_vol) / total_vol * 10  # scale to ~-10 to 10
        else:
            main_force_ratio = 0
        
        return {
            "total_amount": int(np.sum(df["amount"].values[-5:])),
            "main_force": int(main_force_ratio * 1e8),
            "main_force_ratio": round(main_force_ratio, 2),
            "retail": int(-main_force_ratio * 0.3 * 1e8),
            "retail_ratio": round(-main_force_ratio * 0.3, 2),
            "medium": 0,
            "medium_ratio": 0,
            "net_flow": int(main_force_ratio * 0.7 * 1e8),
            "signal": "主力流入" if main_force_ratio > 1 else (
                "主力流出" if main_force_ratio < -1 else "资金平衡"
            ),
        }
    except Exception:
        return _demo_money_flow(symbol)


def _demo_money_flow(symbol: str) -> dict:
    """Demo money flow"""
    np.random.seed(hash(f"{symbol}_flow") % (2**31))
    total_amount = np.random.lognormal(17, 0.5)
    main_force_ratio = np.random.uniform(-0.05, 0.08)
    main_force = total_amount * main_force_ratio
    retail_ratio = np.random.uniform(-0.03, 0.03)
    retail = total_amount * retail_ratio
    net_flow = main_force + retail
    
    return {
        "total_amount": int(total_amount),
        "main_force": int(main_force),
        "main_force_ratio": round(main_force_ratio * 100, 2),
        "retail": int(retail * 0.3),
        "retail_ratio": round(retail_ratio * 100, 2),
        "medium": 0,
        "medium_ratio": 0,
        "net_flow": int(net_flow),
        "signal": "主力流入" if main_force > 0 else "主力流出",
    }


# ==========================================================
#  快速数据验证/摘要
# ==========================================================

def data_quality_check() -> dict:
    """数据质量验证报告"""
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        network_ok = True
    except OSError:
        network_ok = False
    
    # Check if Sina API works
    sina_ok = False
    try:
        r = requests.get(
            "https://hq.sinajs.cn/list=sh600519",
            headers=SINA_HEADERS,
            timeout=5
        )
        sina_ok = r.status_code == 200 and len(r.text) > 50
    except Exception:
        pass
    
    cache_size = 0
    try:
        cache_size = round(
            sum(f.stat().st_size for f in CACHE_DIR.glob("**/*") if f.is_file()) / 1e6, 2
        )
    except Exception:
        pass
    
    return {
        "timestamp": datetime.now().isoformat(),
        "network_available": network_ok,
        "sina_api_available": sina_ok,
        "cache_size_mb": cache_size,
        "stocks_in_pool": len(CORE_STOCKS),
        "industries": len(set(ind for _, _, ind in CORE_STOCKS)),
    }


def scan_market() -> pd.DataFrame:
    """全市场扫描，获取A股实时表现"""
    df = fetch_realtime_quotes()
    if df.empty:
        return df
    
    df["rank_by_change"] = df["change_pct"].rank(ascending=False)
    df["rank_by_volume"] = df["amount"].rank(ascending=False)
    
    if "turnover" in df.columns:
        df["score"] = (
            df["change_pct"].rank(pct=True) * 0.5 +
            df["amount"].rank(pct=True) * 0.3 +
            df["turnover"].rank(pct=True) * 0.2
        )
    else:
        df["score"] = df["change_pct"].rank(pct=True)
    
    return df.sort_values("score", ascending=False)
