"""
Hyperion Pro — 专业投资报告生成器
====================================
生成华尔街级别的投资分析报告

报告类型：
  1. 日报 (Daily Brief) — 每日市场总览 + 操作建议
  2. 精选研报 (Stock Report) — 个股深度分析
  3. 行业轮动报告 (Sector Rotation)
  4. 组合周报 (Portfolio Weekly)
  5. 风险预警报告 (Risk Alert)

每个报告都包含具体可操作的策略建议
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

from ..data.market import (
    fetch_realtime_quotes, fetch_index_quotes, fetch_history,
    sector_quotes, CORE_STOCKS, INDICES, get_stock_name
)
from ..analysis.market_state import MarketStateAnalyzer
from ..analysis.signals import SignalGenerator

REPORTS_DIR = Path.home() / ".hyperion_data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ReportGenerator:
    """投资报告生成器"""
    
    @staticmethod
    def daily_brief() -> str:
        """
        每日投资简报
        
        Returns:
            markdown 格式的完整日报
        """
        now = datetime.now()
        date_str = now.strftime("%Y年%m月%d日")
        weekday = "星期一 星期二 星期三 星期四 星期五 星期六 星期日".split()[now.weekday()]
        
        # 数据采集
        market_outlook = MarketStateAnalyzer.generate_outlook()
        overall = MarketStateAnalyzer.analyze_overall()
        emotion = MarketStateAnalyzer.analyze_emotion()
        sectors = MarketStateAnalyzer.analyze_sectors()
        
        # 信号
        sig_gen = SignalGenerator()
        top_buy = sig_gen.top_buy_signals(10)
        top_sell = sig_gen.top_sell_signals(5)
        sector_recs = sig_gen.sector_recommendations()
        
        # 指数行情
        indices = fetch_index_quotes()
        
        # ========== 构建报告 ==========
        lines = []
        
        # === 头 ===
        lines.append(f"# HYPERION PRO 每日投资简报")
        lines.append(f"")
        lines.append(f"**{date_str} ({weekday})** | 数据时间: {now.strftime('%H:%M')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # === 一、市场总览 ===
        lines.append(f"## 一、市场总览")
        lines.append(f"")
        lines.append(f"### 大盘判断：**{market_outlook.get('market_state', '未知')}**")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 市场情绪 | {emotion.get('emotion', '未知')} (评分: {emotion.get('score', 0)}) |")
        lines.append(f"| 风险等级 | {market_outlook.get('risk_level', '中')} |")
        lines.append(f"| 推荐仓位 | {market_outlook.get('recommended_position', '30%-50%')} |")
        lines.append(f"| 置信度 | {market_outlook.get('confidence', 0) * 100:.0f}% |")
        
        # 指数表现
        lines.append(f"")
        lines.append(f"### 主要指数表现")
        lines.append(f"")
        if not indices.empty:
            lines.append(f"| 指数 | 最新价 | 涨跌幅 |")
            lines.append(f"|------|--------|--------|")
            for _, row in indices.iterrows():
                code = row.get("code", "")
                name = INDICES.get(code, row.get("name", ""))
                price = row.get("price", 0)
                chg = row.get("change_pct", 0)
                arrow = "📈" if chg >= 0 else "📉"
                lines.append(f"| {name} | {price} | {arrow} {chg:+.2f}% |")
        
        # 技术指标
        idx = overall.get("index_status", {})
        lines.append(f"")
        lines.append(f"### 技术指标")
        lines.append(f"- 短期趋势: **{idx.get('short_trend', '未知')}**")
        lines.append(f"- 中期趋势: **{idx.get('mid_trend', '未知')}**")
        lines.append(f"- 均线排列: **{idx.get('ma_alignment', '未知')}**")
        lines.append(f"- 近5日涨幅: {idx.get('short_return', 0):+.2f}%")
        lines.append(f"- 近月涨幅: {idx.get('monthly_return', 0):+.2f}%")
        lines.append(f"- 20日波动率: {idx.get('volatility_20d', 0):.2f}%")
        
        # 量能
        vol = overall.get("volume_analysis", {})
        lines.append(f"- 量能状态: **{vol.get('vol_status', '正常')}**")
        
        # 涨跌家数
        breadth = overall.get("breadth_analysis", {})
        lines.append(f"- 涨跌比: {breadth.get('up_stocks', 0)}/{breadth.get('down_stocks', 0)} ({breadth.get('up_ratio', 0):.1f}%)")
        lines.append(f"- 涨跌家数状态: **{breadth.get('breadth_status', '未知')}**")
        
        lines.append(f"")
        lines.append(f"### 市场情绪")
        lines.append(f"- 情绪判断: **{emotion.get('emotion', '未知')}**")
        lines.append(f"- 涨停家数: {emotion.get('limit_up', 0)}")
        lines.append(f"- 跌停家数: {emotion.get('limit_down', 0)}")
        lines.append(f"- 涨幅中位数: {emotion.get('median_change', 0):+.2f}%")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # === 二、操作策略 ===
        lines.append(f"## 二、操作策略")
        lines.append(f"")
        lines.append(f"**核心建议：** {market_outlook.get('action_advice', '暂无')}")
        lines.append(f"")
        lines.append(f"### 仓位管理")
        lines.append(f"| 项目 | 建议 |")
        lines.append(f"|------|------|")
        lines.append(f"| 总仓位 | {market_outlook.get('recommended_position', '30%-50%')} |")
        lines.append(f"| 单股上限 | 不超过总仓位10% |")
        lines.append(f"| 止损纪律 | 单笔亏损超过5%止损 |")
        lines.append(f"| 风险控制 | {market_outlook.get('risk_level', '中')}风险，注意仓位管理 |")
        
        # 热点板块
        hot = market_outlook.get("hot_sectors", [])
        cold = market_outlook.get("cold_sectors", [])
        if hot:
            lines.append(f"")
            lines.append(f"### 热点板块")
            lines.append(f"关注: **{'、'.join(hot)}**")
        if cold:
            lines.append(f"")
            lines.append(f"### 回避板块")
            lines.append(f"回避: **{'、'.join(cold)}**")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # === 三、行业配置建议 ===
        lines.append(f"## 三、行业配置建议")
        lines.append(f"")
        
        # 按推荐排序
        sorted_sectors = sorted(sector_recs.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        
        lines.append(f"| 行业 | 评分 | 建议 | 推荐标的 |")
        lines.append(f"|------|------|------|----------|")
        for industry, rec in sorted_sectors:
            top_stocks_str = "、".join([f"{s['name']}({s['score']})" for s in rec["top_stocks"]])
            rec_label = rec["recommendation"]
            
            # 推荐标记
            rec_mark = ""
            if rec["avg_score"] > 20:
                rec_mark = "⭐⭐"
            elif rec["avg_score"] > 10:
                rec_mark = "⭐"
            elif rec["avg_score"] < -10:
                rec_mark = "⚠️"
            
            lines.append(f"| {industry} | {rec['avg_score']:+.1f} | {rec_label}{rec_mark} | {top_stocks_str} |")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # === 四、个股推荐 ===
        lines.append(f"## 四、今日精选个股")
        lines.append(f"")
        
        lines.append(f"### 买入信号 (Top 10)")
        lines.append(f"")
        lines.append(f"| 代码 | 名称 | 行业 | 信号 | 评分 | 现价 | 目标价 | 止损价 | 上涨空间 | 理由 |")
        lines.append(f"|------|------|------|------|------|------|--------|--------|----------|------|")
        
        for sig in top_buy:
            lines.append(f"| {sig.code} | {sig.name} | {sig.industry} | **{sig.signal}** | {sig.score:+.1f} | {sig.current_price:.2f} | {sig.target_price:.2f} | {sig.stop_loss:.2f} | {sig.upside_potential:+.1f}% | {sig.reasons[0] if sig.reasons else ''} |")
        
        if top_sell:
            lines.append(f"")
            lines.append(f"### 卖出信号")
            lines.append(f"")
            lines.append(f"| 代码 | 名称 | 行业 | 信号 | 评分 | 现价 | 理由 |")
            lines.append(f"|------|------|------|------|------|------|------|")
            for sig in top_sell:
                lines.append(f"| {sig.code} | {sig.name} | {sig.industry} | **{sig.signal}** | {sig.score:+.1f} | {sig.current_price:.2f} | {sig.reasons[0] if sig.reasons else ''} |")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # === 五、风险提示 ===
        lines.append(f"## 五、风险提示")
        lines.append(f"")
        lines.append(f"1. 以上分析基于技术指标和市场数据，不构成投资建议")
        lines.append(f"2. 股市有风险，投资需谨慎")
        lines.append(f"3. 建议根据自身风险承受能力合理配置仓位")
        lines.append(f"4. 严格设置止损，控制单笔亏损")
        
        if overall.get("risk_level") in ("高", "中偏高"):
            lines.append(f"5. 当前市场**{overall.get('risk_level')}**风险，建议控制仓位在{market_outlook.get('recommended_position', '30%以下')}")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*报告生成时间: {now.strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"*数据来源: 东方财富 / AkShare*")
        
        report = "\n".join(lines)
        
        # 保存
        filename = f"daily_brief_{now.strftime('%Y%m%d')}.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
    
    @staticmethod
    def stock_report(code: str) -> str:
        """
        个股深度分析报告
        
        Args:
            code: 股票代码
            
        Returns:
            markdown 报告
        """
        from ..analysis.technical import TechnicalAnalyzer
        from ..analysis.signals import SignalGenerator
        
        history = fetch_history(code, days=180)
        if history.empty:
            return f"# {code} — 数据不足"
        
        name = get_stock_name(code)
        
        # 技术分析
        analyzer = TechnicalAnalyzer()
        tech = analyzer.comprehensive_analysis(history)
        
        # 信号
        sig_gen = SignalGenerator()
        signal = sig_gen.analyze_stock(code)
        
        # 基本数据
        close = history["close"].values
        high = history["high"].values
        low = history["low"].values
        
        lines = []
        lines.append(f"# {name}({code}) — 深度分析报告")
        lines.append(f"")
        lines.append(f"**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        if signal:
            lines.append(f"## 综合评级: **{signal.signal}** (评分: {signal.score:+.1f})")
            lines.append(f"")
            lines.append(f"| 指标 | 数值 |")
            lines.append(f"|------|------|")
            lines.append(f"| 当前价格 | {signal.current_price:.2f} |")
            lines.append(f"| 目标价 | {signal.target_price:.2f} |")
            lines.append(f"| 止损价 | {signal.stop_loss:.2f} |")
            lines.append(f"| 上涨空间 | {signal.upside_potential:+.1f}% |")
            lines.append(f"| 下跌风险 | {signal.downside_risk:.1f}% |")
            lines.append(f"| 盈亏比 | {signal.reward_risk_ratio:.2f} |")
            lines.append(f"| 置信度 | {signal.confidence * 100:.0f}% |")
            lines.append(f"")
            lines.append(f"**操作建议**: {signal.advice}")
            lines.append(f"")
            if signal.reasons:
                lines.append(f"**推荐理由**:")
                for r in signal.reasons:
                    lines.append(f"- {r}")
                lines.append(f"")
        
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## 技术指标详解")
        lines.append(f"")
        
        # MACD
        macd = tech.get("macd", {})
        lines.append(f"### MACD")
        lines.append(f"- DIF: {macd.get('dif', 'N/A')}")
        lines.append(f"- DEA: {macd.get('dea', 'N/A')}")
        lines.append(f"- 柱状图: {macd.get('histogram', 'N/A')}")
        lines.append(f"- 判断: {macd.get('description', 'N/A')}")
        lines.append(f"- 背离: {macd.get('divergence', '无')}")
        lines.append(f"")
        
        # RSI
        rsi = tech.get("rsi", {})
        lines.append(f"### RSI (14)")
        lines.append(f"- RSI值: {rsi.get('rsi', 'N/A')}")
        lines.append(f"- 判断: {rsi.get('description', 'N/A')}")
        lines.append(f"- 操作: {rsi.get('action', 'N/A')}")
        lines.append(f"")
        
        # KDJ
        kdj = tech.get("kdj", {})
        lines.append(f"### KDJ")
        lines.append(f"- K: {kdj.get('k', 'N/A')}, D: {kdj.get('d', 'N/A')}, J: {kdj.get('j', 'N/A')}")
        lines.append(f"- 信号: {kdj.get('signal', 'N/A')}")
        lines.append(f"- 操作: {kdj.get('action', 'N/A')}")
        lines.append(f"")
        
        # 布林带
        boll = tech.get("boll", {})
        lines.append(f"### 布林带 (20,2)")
        lines.append(f"- 上轨: {boll.get('upper', 'N/A')}")
        lines.append(f"- 中轨: {boll.get('middle', 'N/A')}")
        lines.append(f"- 下轨: {boll.get('lower', 'N/A')}")
        lines.append(f"- 带宽: {boll.get('bandwidth', 'N/A')}%")
        lines.append(f"- 位置: {boll.get('position', 'N/A')}%")
        lines.append(f"- 判断: {boll.get('signal', 'N/A')}")
        lines.append(f"- 操作: {boll.get('action', 'N/A')}")
        lines.append(f"")
        
        # 均线
        ma = tech.get("ma", {})
        lines.append(f"### 均线系统")
        lines.append(f"- MA5: {ma.get('ma5', 'N/A')}")
        lines.append(f"- MA10: {ma.get('ma10', 'N/A')}")
        lines.append(f"- MA20: {ma.get('ma20', 'N/A')}")
        lines.append(f"- MA60: {ma.get('ma60', 'N/A')}")
        lines.append(f"- 排列: {ma.get('alignment', 'N/A')}")
        lines.append(f"- 操作: {ma.get('description', 'N/A')}")
        lines.append(f"")
        
        # 成交量
        volume = tech.get("volume", {})
        lines.append(f"### 成交量分析")
        lines.append(f"- 当日量: {volume.get('current_volume', 'N/A')}")
        lines.append(f"- 5日均量: {volume.get('ma5_volume', 'N/A')}")
        lines.append(f"- 量比(5日): {volume.get('vol_ratio_5d', 'N/A')}")
        lines.append(f"- 量价关系: {volume.get('vp_relation', 'N/A')}")
        lines.append(f"")
        
        # 支撑阻力
        sr = tech.get("sr", {})
        lines.append(f"### 支撑阻力位")
        lines.append(f"- 近期高点: {sr.get('recent_high', 'N/A')}")
        lines.append(f"- 近期低点: {sr.get('recent_low', 'N/A')}")
        lines.append(f"- 最近支撑: {sr.get('nearest_support', 'N/A')}")
        lines.append(f"- 最近阻力: {sr.get('nearest_resistance', 'N/A')}")
        lines.append(f"- 距支撑: {sr.get('dist_to_support', 'N/A')}%")
        lines.append(f"- 距阻力: {sr.get('dist_to_resistance', 'N/A')}%")
        lines.append(f"- 突破信号: {sr.get('break_signal', 'N/A')}")
        lines.append(f"")
        
        # 综合评分
        comp = tech.get("comprehensive", {})
        lines.append(f"### 综合评分")
        lines.append(f"- 总分: {comp.get('total_score', 'N/A')}分")
        lines.append(f"- 信号: {comp.get('signal', 'N/A')}")
        lines.append(f"- 建议: {comp.get('advice', 'N/A')}")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*本报告由 Hyperion Pro 量化系统自动生成*")
        lines.append(f"*数据区间: {history['date'].iloc[0].strftime('%Y-%m-%d') if 'date' in history.columns else 'N/A'} ~ {datetime.now().strftime('%Y-%m-%d')}*")
        
        report = "\n".join(lines)
        
        # 保存
        filename = f"stock_report_{code}_{datetime.now().strftime('%Y%m%d')}.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
    
    @staticmethod
    def portfolio_report(portfolio: Dict[str, float]) -> str:
        """
        组合分析报告
        
        Args:
            portfolio: {code: weight} 当前持仓
        """
        sig_gen = SignalGenerator()
        
        lines = []
        lines.append(f"# 投资组合分析报告")
        lines.append(f"")
        lines.append(f"**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**持仓数量**: {len(portfolio)} 只")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        lines.append(f"## 持仓分析")
        lines.append(f"")
        lines.append(f"| 代码 | 名称 | 行业 | 仓位 | 信号 | 评分 | 操作建议 |")
        lines.append(f"|------|------|------|------|------|------|----------|")
        
        total_score = 0
        hold_count = 0
        sell_count = 0
        buy_count = 0
        
        for code, weight in portfolio.items():
            sig = sig_gen.analyze_stock(code)
            if sig:
                total_score += sig.score * weight
                hold_count += 1
                if "卖出" in sig.signal:
                    sell_count += 1
                elif "买入" in sig.signal:
                    buy_count += 1
                
                action = ""
                if "卖出" in sig.signal:
                    action = "⚠️ 建议减仓"
                elif "买入" in sig.signal:
                    action = "✅ 建议持有"
                else:
                    action = "⏸️ 观望"
                
                lines.append(f"| {code} | {sig.name} | {sig.industry} | {weight*100:.0f}% | {sig.signal} | {sig.score:+.1f} | {action} |")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## 组合诊断")
        lines.append(f"")
        lines.append(f"- 综合评分: {total_score:+.1f}")
        lines.append(f"- 建议卖出: {sell_count} 只")
        lines.append(f"- 建议买入/持有: {buy_count} 只")
        lines.append(f"- 观望: {hold_count - sell_count - buy_count} 只")
        
        if sell_count > hold_count / 2:
            lines.append(f"- ⚠️ **预警**: 组合中超过半数标的技术面恶化")
        
        lines.append(f"")
        lines.append(f"*本报告由 Hyperion Pro 量化系统自动生成*")
        
        report = "\n".join(lines)
        
        filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
    
    @staticmethod
    def risk_alert() -> str:
        """
        风险预警报告
        
        Returns:
            预警内容
        """
        overall = MarketStateAnalyzer.analyze_overall()
        
        alerts = []
        risk_level = overall.get("risk_level", "低")
        
        if risk_level in ("高", "中偏高"):
            alerts.append(f"⚡ **市场风险预警**: 当前市场处于**{risk_level}**风险状态")
            alerts.append(f"")
            alerts.append(f"| 预警项目 | 状态 | 说明 |")
            alerts.append(f"|----------|------|------|")
            
            idx = overall.get("index_status", {})
            trend = idx.get("short_trend", "")
            alignment = idx.get("ma_alignment", "")
            
            if trend == "向下":
                alerts.append(f"| 短期趋势 | ⚠️ 下行 | 5日线低于10日线 |")
            if "空头" in alignment:
                alerts.append(f"| 均线排列 | ⚠️ 空头 | 短期均线低于长期均线 |")
            
            breadth = overall.get("breadth_analysis", {})
            up_ratio = breadth.get("up_ratio", 50)
            if up_ratio < 30:
                alerts.append(f"| 涨跌比 | ⚠️ 弱势 | 仅{up_ratio:.0f}%个股上涨 |")
            
            alerts.append(f"")
            alerts.append(f"**建议操作**:")
            alerts.append(f"1. 降低总仓位至{overall.get('recommended_position', '30%以下')}")
            alerts.append(f"2. 减少高Beta个股持仓")
            alerts.append(f"3. 增加防御性配置 (公共事业、消费等)")
            alerts.append(f"4. 严格止损，控制回撤")
        else:
            alerts.append(f"✅ 当前市场风险可控")
        
        report = "\n".join(alerts)
        
        filename = f"risk_alert_{datetime.now().strftime('%Y%m%d')}.md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        return report
    
    @staticmethod
    def generate_all() -> dict:
        """生成所有报告"""
        return {
            "daily_brief": ReportGenerator.daily_brief(),
            "risk_alert": ReportGenerator.risk_alert(),
        }
    
    @staticmethod
    def list_reports() -> List[dict]:
        """列出所有已生成的报告"""
        reports = []
        for f in sorted(REPORTS_DIR.glob("*.md"), reverse=True):
            reports.append({
                "filename": f.name,
                "size_kb": f.stat().st_size / 1024,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "path": str(f),
            })
        return reports
