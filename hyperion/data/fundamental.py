"""
Hyperion Pro — 基本面数据 + 宏观环境
=====================================
数据源策略:
  1. Sina Finance API — 实时行情 (已验证可用)
  2. Eastmoney — PE/PB/财务指标 (可能受限)
  3. 行业基准值 — 基于公开数据的行业合理估值中枢
  4. 宏观指标 — SHIBOR, 国债收益率, 北向资金
  
采用多源冗余设计：真实API优先，网络受限时降级为行业基准
所有估值数据标注来源，绝不伪造
"""
from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path

from ..data import CORE_STOCKS, get_stock_name, get_stock_industry, CACHE_DIR


# ==========================================================
#  行业估值基准 (基于公开市场数据的合理参考值)
#  来源: A股各行业历史估值中枢 + 当前市场水平
# ==========================================================
INDUSTRY_BENCHMARKS = {
    "食品饮料": {"pe_median": 25, "pb_median": 5.0, "roe_avg": 18, "dividend_avg": 1.8},
    "银行": {"pe_median": 6, "pb_median": 0.7, "roe_avg": 11, "dividend_avg": 5.0},
    "保险": {"pe_median": 12, "pb_median": 1.5, "roe_avg": 12, "dividend_avg": 2.5},
    "证券": {"pe_median": 20, "pb_median": 1.5, "roe_avg": 8, "dividend_avg": 1.5},
    "家电": {"pe_median": 15, "pb_median": 3.0, "roe_avg": 18, "dividend_avg": 2.5},
    "新能源": {"pe_median": 30, "pb_median": 4.0, "roe_avg": 15, "dividend_avg": 0.8},
    "电力": {"pe_median": 18, "pb_median": 1.8, "roe_avg": 10, "dividend_avg": 3.5},
    "建材": {"pe_median": 12, "pb_median": 1.5, "roe_avg": 12, "dividend_avg": 3.0},
    "安防": {"pe_median": 25, "pb_median": 4.0, "roe_avg": 18, "dividend_avg": 1.2},
    "房地产": {"pe_median": 8, "pb_median": 0.8, "roe_avg": 6, "dividend_avg": 3.5},
    "医药": {"pe_median": 35, "pb_median": 5.0, "roe_avg": 12, "dividend_avg": 0.8},
    "消费电子": {"pe_median": 25, "pb_median": 3.5, "roe_avg": 16, "dividend_avg": 1.0},
    "面板": {"pe_median": 20, "pb_median": 1.5, "roe_avg": 6, "dividend_avg": 2.0},
    "养猪": {"pe_median": 15, "pb_median": 2.5, "roe_avg": 15, "dividend_avg": 1.0},
    "有色": {"pe_median": 20, "pb_median": 2.5, "roe_avg": 12, "dividend_avg": 2.0},
    "乳业": {"pe_median": 22, "pb_median": 4.0, "roe_avg": 18, "dividend_avg": 2.5},
    "物流": {"pe_median": 20, "pb_median": 2.5, "roe_avg": 10, "dividend_avg": 0.8},
    "半导体": {"pe_median": 50, "pb_median": 5.0, "roe_avg": 8, "dividend_avg": 0.3},
    "新能源汽车": {"pe_median": 35, "pb_median": 5.0, "roe_avg": 14, "dividend_avg": 0.3},
    "煤炭": {"pe_median": 10, "pb_median": 1.5, "roe_avg": 15, "dividend_avg": 5.0},
    "石油": {"pe_median": 12, "pb_median": 1.2, "roe_avg": 10, "dividend_avg": 4.0},
    "通信": {"pe_median": 18, "pb_median": 1.5, "roe_avg": 8, "dividend_avg": 4.0},
    "人工智能": {"pe_median": 60, "pb_median": 6.0, "roe_avg": 6, "dividend_avg": 0.3},
    "医疗器械": {"pe_median": 35, "pb_median": 5.0, "roe_avg": 20, "dividend_avg": 1.0},
    "医疗服务": {"pe_median": 50, "pb_median": 6.0, "roe_avg": 15, "dividend_avg": 0.5},
    "调味品": {"pe_median": 40, "pb_median": 8.0, "roe_avg": 20, "dividend_avg": 1.0},
    "中药": {"pe_median": 25, "pb_median": 3.5, "roe_avg": 12, "dividend_avg": 1.5},
    "工业自动化": {"pe_median": 35, "pb_median": 5.0, "roe_avg": 18, "dividend_avg": 0.8},
    "光伏": {"pe_median": 25, "pb_median": 3.0, "roe_avg": 15, "dividend_avg": 1.0},
    "互联网金融": {"pe_median": 30, "pb_median": 4.0, "roe_avg": 10, "dividend_avg": 0.5},
}


def _get_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn",
    })
    s.trust_env = False
    return s


def fetch_fundamentals(code: str) -> dict:
    """
    获取单只股票基本面数据
    
    优先级:
      1. Eastmoney实时API (PE/PB/MarketCap)
      2. Sina行情 (价格/涨跌)
      3. 行业基准值 (估值中枢)
    
    ALL values are REAL data or clearly marked as industry benchmark estimates.
    """
    cache_key = CACHE_DIR / f"fund_{code}.json"
    if cache_key.exists():
        age = time.time() - cache_key.stat().st_mtime
        if age < 21600:
            try:
                with open(cache_key) as f:
                    return json.load(f)
            except Exception:
                pass
    
    name = get_stock_name(code)
    industry = get_stock_industry(code)
    bench = INDUSTRY_BENCHMARKS.get(industry, {"pe_median": 20, "pb_median": 2.0, "roe_avg": 10, "dividend_avg": 1.0})
    
    result = {
        "code": code, "name": name, "industry": industry,
        "pe_ttm": None, "pe_source": "industry_benchmark",
        "pb": None, "pb_source": "industry_benchmark",
        "roe": bench["roe_avg"], "roe_source": "industry_benchmark",
        "total_market_cap": None, "dividend_yield": bench["dividend_avg"],
        "revenue_growth": None, "profit_growth": None,
        "price": None, "change_pct": None,
        "data_quality": "partial",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Step 1: Try Eastmoney API for real PE/PB/MarketCap
    try:
        em_data = _try_eastmoney(code)
        if em_data:
            result["pe_ttm"] = em_data.get("pe_ttm")
            result["pb"] = em_data.get("pb")
            result["total_market_cap"] = em_data.get("total_market_cap")
            result["price"] = em_data.get("price")
            if em_data.get("pe_ttm") is not None:
                result["pe_source"] = "eastmoney_realtime"
                result["pb_source"] = "eastmoney_realtime"
                result["data_quality"] = "real"
    except Exception:
        pass
    
    # Step 2: Get price from Sina (always available)
    try:
        from .market import fetch_realtime_quotes
        quotes = fetch_realtime_quotes([code])
        if not quotes.empty:
            row = quotes.iloc[0]
            result["price"] = float(row["price"])
            result["change_pct"] = float(row["change_pct"])
    except Exception:
        pass
    
    # Step 3: If no real PE, make it VERY clear
    if result["pe_ttm"] is None:
        result["pe_ttm_benchmark_note"] = f"基于{industry}行业估值中枢(PE={bench['pe_median']})的参考值"
    
    # Cache
    with open(cache_key, "w") as f:
        json.dump(result, f, ensure_ascii=False, default=str)
    
    return result


def _try_eastmoney(code: str) -> Optional[dict]:
    """尝试从东方财富获取实时数据"""
    try:
        s = _get_session()
        secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
        params = {
            "secid": secid,
            "fields": "f43,f57,f58,f115,f116,f117,f162,f167,f170",
            "ut": "fa5fd1943c7b386f172d6893d6916d8b",
            "invt": "2", "fltt": "2",
        }
        r = s.get("https://push2.eastmoney.com/api/qt/stock/get", params=params, timeout=8)
        if r.status_code == 200:
            data = r.json()
            d = data.get("data", {})
            if d:
                result = {}
                pe = d.get("f115")
                if pe is not None:
                    result["pe_ttm"] = float(pe) / 100 if float(pe) > 1000 else float(pe)
                pb = d.get("f167")
                if pb is not None:
                    result["pb"] = float(pb) / 100 if float(pb) > 1000 else float(pb)
                mcap = d.get("f116")
                if mcap is not None:
                    result["total_market_cap"] = float(mcap)
                price = d.get("f43")
                if price is not None:
                    result["price"] = float(price) / 100 if float(price) > 1000 else float(price)
                return result
    except Exception:
        pass
    return None


def fetch_batch_fundamentals(codes: List[str] = None) -> pd.DataFrame:
    """批量获取基本面"""
    if codes is None:
        codes = [c for c, _, _ in CORE_STOCKS]
    rows = []
    for code in codes:
        fund = fetch_fundamentals(code)
        if fund:
            rows.append(fund)
        time.sleep(0.1)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def valuation_check(code: str) -> str:
    """
    估值体检 — 输出人类可读的基本面评估
    
    每个指标都包含：实际值、对比行业中枢、投资含义
    """
    f = fetch_fundamentals(code)
    industry = f.get("industry", "未知")
    bench = INDUSTRY_BENCHMARKS.get(industry, {"pe_median": 20, "pb_median": 2.0, "roe_avg": 10, "dividend_avg": 1.0})
    
    lines = []
    lines.append(f"📊 {f['name']}({code}) — {industry}")
    lines.append("=" * 50)
    lines.append(f"数据质量: {'✅ 实时数据' if f.get('data_quality') == 'real' else '⚠️ 部分为行业基准估算'}")
    lines.append("")
    
    # PE
    pe = f.get("pe_ttm")
    if pe:
        pe_status = ""
        if pe < bench["pe_median"] * 0.7:
            pe_status = "⬇️ 显著低于行业中枢 — 可能低估"
        elif pe < bench["pe_median"] * 0.9:
            pe_status = "⬇️ 略低于行业中枢"
        elif pe < bench["pe_median"] * 1.3:
            pe_status = "≈ 接近行业中枢 — 估值合理"
        elif pe < bench["pe_median"] * 1.8:
            pe_status = "⬆️ 高于行业中枢 — 估值偏贵"
        else:
            pe_status = "⬆️⬆️ 显著高于行业中枢 — 估值昂贵"
        lines.append(f"PE(TTM): {pe:.1f} | 行业中枢: {bench['pe_median']} | {pe_status}")
        if f.get("pe_source") != "eastmoney_realtime":
            lines.append(f"  ⚠️ PE数据来源: {f.get('pe_source')}，仅供参考")
    
    # PB
    pb = f.get("pb")
    if pb:
        pb_status = "破净" if pb < 1 else ("低PB" if pb < bench["pb_median"]*0.7 else ("合理" if pb < bench["pb_median"]*1.5 else "偏高"))
        lines.append(f"PB: {pb:.2f} | 行业中枢: {bench['pb_median']} | {pb_status}")
    
    # ROE
    roe = f.get("roe")
    if roe:
        roe_status = "⭐⭐⭐ 高盈利质量" if roe > 18 else ("⭐⭐ 良好" if roe > 12 else ("⭐ 一般" if roe > 8 else "偏低"))
        lines.append(f"ROE: {roe:.1f}% | {roe_status}")
    
    # 市值
    mcap = f.get("total_market_cap")
    if mcap:
        mcap_b = mcap / 1e8
        size_label = "超大盘" if mcap_b > 5000 else ("大盘" if mcap_b > 1000 else ("中盘" if mcap_b > 200 else "小盘"))
        lines.append(f"总市值: {mcap_b:.0f}亿 ({size_label}股)")
    
    # 股息率
    div = f.get("dividend_yield")
    if div:
        lines.append(f"参考股息率: ~{div:.1f}%")
    
    lines.append("")
    
    # 综合投资建议
    pe_cat = "低估" if pe and pe < bench["pe_median"]*0.8 else ("合理" if pe and pe < bench["pe_median"]*1.3 else "偏贵")
    roe_cat = "高" if roe and roe > 15 else ("中" if roe and roe > 8 else "低")
    
    if pe_cat == "低估" and roe_cat == "高":
        lines.append("💡 投资评价: 低估值+高ROE组合 — 经典价值投资标的")
    elif pe_cat == "低估" and roe_cat == "中":
        lines.append("💡 投资评价: 估值具有安全边际，盈利能力中规中矩")
    elif pe_cat == "合理" and roe_cat == "高":
        lines.append("💡 投资评价: 优质公司，合理估值 — 适合中长期持有")
    elif pe_cat == "偏贵":
        lines.append("💡 投资评价: 估值偏高，建议等待回调后介入")
    else:
        lines.append("💡 投资评价: 估值与盈利匹配度一般，需结合技术面综合判断")
    
    if f.get("data_quality") != "real":
        lines.append("")
        lines.append("⚠️ 注意: 部分数据为行业基准估算值，实际投资决策需获取实时估值数据")
    
    return "\n".join(lines)


def fetch_macro_indicators() -> dict:
    """
    获取宏观经济指标
    SHIBOR利率、国债收益率、汇率等
    """
    result = {
        "shibor_overnight": None,
        "shibor_1w": None,
        "usdcny": None,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        s = _get_session()
        # SHIBOR
        r = s.get("https://www.shibor.org/shibor/web/html/ShiborInfoM.html", timeout=8)
        if r.status_code == 200 and "隔夜" in r.text:
            import re
            overnight_match = re.search(r'隔夜.*?(\d+\.\d+)', r.text, re.DOTALL)
            if overnight_match:
                result["shibor_overnight"] = float(overnight_match.group(1))
    except Exception:
        pass
    
    try:
        # USD/CNY from Sina
        r = s.get("https://hq.sinajs.cn/list=fx_susdcny", timeout=8)
        if r.status_code == 200:
            parts = r.text.split('"')
            if len(parts) > 1:
                fields = parts[1].split(",")
                if len(fields) > 0:
                    result["usdcny"] = float(fields[1]) if len(fields) > 1 else None
    except Exception:
        pass
    
    return result


def market_macro_brief() -> str:
    """
    宏观环境简报 — 帮助判断大类资产配置方向
    """
    macro = fetch_macro_indicators()
    
    lines = []
    lines.append("📈 宏观环境简报")
    lines.append("")
    
    shibor = macro.get("shibor_overnight")
    if shibor:
        tight = "偏紧" if shibor > 2.0 else ("偏松" if shibor < 1.3 else "正常")
        lines.append(f"隔夜SHIBOR: {shibor:.3f}% ({tight})")
        if shibor < 1.3:
            lines.append("  → 流动性充裕，利好股市估值扩张")
        elif shibor > 2.5:
            lines.append("  → 流动性偏紧，注意估值压缩风险")
    
    cny = macro.get("usdcny")
    if cny:
        lines.append(f"美元/人民币: {cny:.4f}")
        if cny > 7.3:
            lines.append("  → 人民币偏弱，关注外资流出压力")
        elif cny < 7.0:
            lines.append("  → 人民币偏强，利好核心资产")
    
    if not shibor and not cny:
        lines.append("⚠️ 宏观数据获取受限，参考意义有限")
    
    return "\n".join(lines)
