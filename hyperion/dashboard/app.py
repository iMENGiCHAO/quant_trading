"""
Hyperion Pro — 量化交易仪表盘
==============================
Bloomberg-terminal 级别专业面板

功能:
  1. 市场总览 — 指数/情绪/状态三连卡
  2. 投资决策 — 买入信号+风险预警+操作计划
  3. 策略回测 — 夏普/胜率/最大回撤对比
  4. 绩效追踪 — 信号命中率/历史业绩
  5. 风险管理 — VaR/压力测试/分散化
  6. 报告中心 — 日报/预警/绩效

数据完整性:
  - 所有数据来自真实API (Sina/akShare)
  - 绝不静默降级返回假数据
  - 数据新鲜度实时显示
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hyperion.data.market import (
    fetch_realtime_quotes, fetch_index_quotes, fetch_history,
    sector_quotes, scan_market, money_flow, DataUnavailableError,
    CORE_STOCKS, INDICES, get_stock_name, get_stock_industry
)
from hyperion.analysis.market_state import MarketStateAnalyzer
from hyperion.analysis.technical import TechnicalAnalyzer
from hyperion.analysis.decision_engine import InvestmentDecisionEngine
from hyperion.risk import RiskManager
from hyperion.performance.tracker import PerformanceTracker
from hyperion.analysis.signal_alerts import SignalAlertSystem, AlertLevel
from hyperion.analysis.trade_journal import TradeJournal
from hyperion.reporting.report_generator import ReportGenerator

# ── Dash imports ──
try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, callback
    import plotly.graph_objs as go
    import plotly.express as px
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

# ── 设计系统 ──
class Theme:
    BG           = "#060B14"
    CARD         = "#0D1525"
    CARD_HOVER   = "#1A2332"
    BORDER       = "#1A2A3D"
    DIVIDER      = "#1A2636"
    TEXT         = "#E8EDF2"
    TEXT_SEC     = "#7B8CA3"
    ACCENT       = "#38BDF8"
    ACCENT_GLOW  = "rgba(56,189,248,0.25)"
    GREEN        = "#10B981"
    GREEN_GLOW   = "rgba(16,185,129,0.25)"
    RED          = "#EF4444"
    RED_GLOW     = "rgba(239,68,68,0.25)"
    AMBER        = "#F59E0B"
    AMBER_GLOW   = "rgba(245,158,11,0.25)"
    PURPLE       = "#8B5CF6"
    CYAN         = "#06B6D4"
    
    @staticmethod
    def px_colors():
        return [Theme.ACCENT, Theme.GREEN, Theme.AMBER, Theme.PURPLE, Theme.CYAN, Theme.RED]

T = Theme

# 布局常量
CARD    = "background:linear-gradient(135deg, #0D1525 0%, #0A1020 100%); border:1px solid #1A2A3D; border-radius:14px; padding:24px; backdrop-filter:blur(10px)"
GLOW_ACCENT = "box-shadow:0 0 40px rgba(56,189,248,0.15), inset 0 1px 0 rgba(255,255,255,0.03)"
GLOW_GREEN  = "box-shadow:0 0 40px rgba(16,185,129,0.15), inset 0 1px 0 rgba(255,255,255,0.03)"
GLOW_RED    = "box-shadow:0 0 40px rgba(239,68,68,0.15), inset 0 1px 0 rgba(255,255,255,0.03)"

CHART_LAYOUT = dict(
    paper_bgcolor=T.BG, plot_bgcolor=T.CARD,
    font=dict(color=T.TEXT, size=12, family="'Inter', 'SF Pro Display', -apple-system, sans-serif"),
    xaxis=dict(gridcolor=T.DIVIDER, zeroline=False),
    yaxis=dict(gridcolor=T.DIVIDER, zerolinecolor=T.DIVIDER),
    margin=dict(l=20, r=20, t=30, b=20),
    hovermode="x unified",
)


def metric_card(label, value, subtitle="", color=T.ACCENT, glow="", large=False):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": T.TEXT_SEC, "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "36px" if large else "28px", "fontWeight": "800", "color": color, "lineHeight": "1.15", "letterSpacing": "-0.5px", "fontFamily": "'Inter', 'SF Pro Display', -apple-system, sans-serif"}),
        html.Div(subtitle, style={"fontSize": "12px", "color": T.TEXT_SEC, "marginTop": "6px", "fontWeight": "500"}) if subtitle else None,
    ], style=f"{CARD}; {glow}; flex:1; margin:0 8px")


def section_header(title, count=""):
    return html.Div([
        html.Span(title, style={"fontSize":"14px","fontWeight":"700","color":T.TEXT}),
        html.Span(f" {count}", style={"fontSize":"12px","color":T.TEXT_SEC}) if count else None,
    ], style={"marginBottom":"14px","paddingBottom":"10px","borderBottom":f"1px solid {T.DIVIDER}"})


# ── 信号卡片组件 ──
def buy_signal_card(d):
    score_color = T.GREEN if d.composite_score > 50 else T.AMBER
    ret_str = f"+{d.expected_return:.1f}%" if d.expected_return > 0 else f"{d.expected_return:.1f}%"
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span(d.name, style={"fontSize":"16px","fontWeight":"800","color":T.TEXT,"letterSpacing":"-0.2px"}),
                html.Span(f" {d.code}", style={"fontSize":"11px","color":T.TEXT_SEC,"marginLeft":"6px"}),
            ]),
            html.Div([
                html.Span(d.decision, style={
                    "fontSize":"11px","fontWeight":"700","color":score_color,
                    "padding":"2px 10px","borderRadius":"12px",
                    "background":f"{score_color}18","border":f"1px solid {score_color}30",
                }),
            ]),
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"10px"}),
        
        # Score bar
        html.Div([
            html.Div(style={
                "width":f"{max(4, min(100, (d.composite_score+100)/2))}%",
                "height":"3px","borderRadius":"2px",
                "background":f"linear-gradient(90deg, {T.RED}, {T.AMBER}, {T.GREEN})",
            })
        ], style={"background":T.BORDER,"borderRadius":"2px","marginBottom":"10px"}),
        
        html.Div([
            html.Span(f"¥{d.current_price:.2f}", style={"fontSize":"13px","color":T.TEXT,"fontWeight":"600"}),
            html.Span(f" → ¥{d.target_price_base:.2f}", style={"fontSize":"13px","color":T.ACCENT,"marginLeft":"4px"}),
            html.Span(f" | 止损 ¥{d.stop_loss:.2f}", style={"fontSize":"12px","color":T.RED,"marginLeft":"8px"}),
            html.Span(f" | {ret_str}", style={"fontSize":"12px","color":T.GREEN,"marginLeft":"8px","fontWeight":"600"}),
        ]),
        
        html.Div([
            html.Span(f"盈亏比 {d.reward_risk_ratio:.1f}:1", style={"fontSize":"11px","color":T.TEXT_SEC,"marginRight":"12px"}),
            html.Span(f"仓位 {d.max_position_pct:.0f}%", style={"fontSize":"11px","color":T.PURPLE,"marginRight":"12px"}),
            html.Span(f"{d.holding_period}", style={"fontSize":"11px","color":T.TEXT_SEC}),
            html.Span(f" 置信度 {d.confidence*100:.0f}%", style={"fontSize":"11px","color":T.ACCENT,"marginLeft":"auto"}),
        ], style={"marginTop":"6px","display":"flex"}),
        
        html.Div(d.summary, style={
            "fontSize":"11px","color":T.TEXT_SEC,"marginTop":"8px","padding":"8px 10px",
            "background":"#0A0E17","borderRadius":"6px","lineHeight":"1.5"
        }),
    ], style={
        "background":"linear-gradient(135deg, #0D1525 0%, #0A1020 100%)","border":f"1px solid {T.BORDER}","borderRadius":"14px",
        "padding":"16px"
    })


def risk_card(d):
    return html.Div([
        html.Div([
            html.Span(d.name, style={"fontSize":"14px","fontWeight":"600","color":T.RED}),
            html.Span(f" {d.code}", style={"fontSize":"11px","color":T.TEXT_SEC}),
            html.Span(f" {d.decision}", style={"fontSize":"11px","fontWeight":"600","color":T.RED,"float":"right"}),
        ]),
        html.Div([
            html.Span(f"评分 {d.composite_score:+.0f}", style={"fontSize":"12px","color":T.RED}),
            html.Span(f" | 止损 ¥{d.stop_loss:.2f}", style={"fontSize":"12px","color":T.TEXT_SEC}),
        ], style={"marginTop":"6px"}),
        html.Div(d.summary, style={"fontSize":"11px","color":T.TEXT_SEC,"marginTop":"6px","lineHeight":"1.4"}),
    ], style={"padding":"14px 18px","background":f"{T.RED}0A","border":f"1px solid {T.RED}18","borderRadius":"12px"})


# ═════════════════════════════════════════════════════════════
#  创建 Dash 应用
# ═════════════════════════════════════════════════════════════

def create_app():
    if not HAS_DASH:
        print("需要安装: pip install dash plotly")
        return None
    
    app = dash.Dash(
        __name__,
        title="Hyperion Pro — 量化交易系统",
        update_title=None,
        suppress_callback_exceptions=True,
    )
    
    app.index_string = f"""<!DOCTYPE html>
<html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{ %css%}}
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;background:{T.BG};color:{T.TEXT}}}
::-webkit-scrollbar{{width:6px;height:6px}}
::-webkit-scrollbar-track{{background:{T.BG}}}
::-webkit-scrollbar-thumb{{background:{T.BORDER};border-radius:3px}}
.Select-control{{background:{T.CARD}!important;border-color:{T.BORDER}!important;color:{T.TEXT}!important}}
.Select-menu-outer{{background:{T.CARD}!important;border-color:{T.BORDER}!important}}
.Select-option{{color:{T.TEXT}!important}}
.Select-option.is-focused{{background:{T.CARD_HOVER}!important}}
.dash-tab{{background:{T.CARD}!important;color:{T.TEXT_SEC}!important;border:none!important;padding:10px 24px!important;font-size:13px!important;font-weight:500!important}}
.dash-tab--selected{{background:{T.CARD}!important;color:{T.ACCENT}!important;border-bottom:2px solid {T.ACCENT}!important;font-weight:600!important}}
</style></head>
<body>{{%app_entry%}}<footer>{{%config%}}{{ %scripts%}}{{ %renderer%}}</footer></body></html>"""
    
    # ── 布局 ──
    app.layout = html.Div([
        dcc.Interval(id="refresh", interval=300000),
        dcc.Store(id="store-data"),
        
        # 顶部
        html.Div([
            html.Div([
                html.Span("HYPERION", style={"fontSize":"20px","fontWeight":"800","color":T.ACCENT,"letterSpacing":"1px"}),
                html.Span(" PRO", style={"fontSize":"20px","fontWeight":"300","color":T.TEXT}),
                html.Span(" · 量化交易系统", style={"fontSize":"13px","color":T.TEXT_SEC,"marginLeft":"12px","fontWeight":"400"}),
            ]),
            html.Div(id="header-time", style={"fontSize":"12px","color":T.TEXT_SEC}),
        ], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
                    "padding":"16px 28px","background":T.CARD,"borderBottom":f"1px solid {T.BORDER}"}),
        
        # Tab导航
        dcc.Tabs(id="tabs", value="tab-overview", children=[
            dcc.Tab(label="市场总览", value="tab-overview"),
            dcc.Tab(label="投资决策", value="tab-decisions"),
            dcc.Tab(label="策略回测", value="tab-backtest"),
            dcc.Tab(label="绩效追踪", value="tab-performance"),
            dcc.Tab(label="风险管理", value="tab-risk"),
            dcc.Tab(label="报告中心", value="tab-reports"),
            dcc.Tab(label="信号预警", value="tab-alerts"),
            dcc.Tab(label="交易日志", value="tab-journal"),
        ]),
        
        html.Div(id="tab-content", style={"padding":"24px 28px","minHeight":"calc(100vh - 130px)"}),
    ])
    
    # ── Callbacks ──
    @app.callback(
        Output("header-time", "children"),
        Input("refresh", "n_intervals"),
    )
    def update_time(_):
        return f"数据更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
    )
    def render_tab(tab):
        try:
            if tab == "tab-overview":
                return render_overview()
            elif tab == "tab-decisions":
                return render_decisions()
            elif tab == "tab-backtest":
                return render_backtest()
            elif tab == "tab-performance":
                return render_performance()
            elif tab == "tab-risk":
                return render_risk()
            elif tab == "tab-reports":
                return render_reports()
            elif tab == "tab-alerts":
                return render_alerts()
            elif tab == "tab-journal":
                return render_journal()
        except DataUnavailableError as e:
            return html.Div([
                html.Div("⚠️ 数据获取失败", style={"fontSize":"20px","color":T.AMBER,"marginBottom":"12px"}),
                html.Div(str(e), style={"color":T.TEXT_SEC,"fontSize":"14px"}),
                html.Div("请检查网络连接或稍后重试", style={"color":T.TEXT_SEC,"fontSize":"13px","marginTop":"8px"}),
            ], style={"padding":"60px","textAlign":"center"})
        except Exception as e:
            return html.Div(f"加载失败: {e}", style={"color":T.RED,"padding":"20px"})
        return html.Div()
    
    return app


# ═════════════════════════════════════════════════════════════
#  市场总览页
# ═════════════════════════════════════════════════════════════

def render_overview():
    overall = MarketStateAnalyzer.analyze_overall()
    emotion = MarketStateAnalyzer.analyze_emotion()
    indices = fetch_index_quotes()
    quotes = fetch_realtime_quotes()
    outlook = MarketStateAnalyzer.generate_outlook()
    
    state = overall.get("market_state", "不明朗")
    state_color = { "牛市":T.GREEN, "反弹":T.CYAN, "震荡":T.AMBER, "回调":T.RED, "熊市":T.RED }.get(state, T.TEXT_SEC)
    risk_level = overall.get("risk_level", "中")
    risk_color = { "低":T.GREEN, "中":T.AMBER, "高":T.RED, "中偏高":T.RED }.get(risk_level, T.TEXT_SEC)
    
    return html.Div([
        # ── 第一行: 市场状态四连卡 ──
        html.Div([
            metric_card("市场状态", state, f"置信度 {overall.get('confidence',0)*100:.0f}%", state_color, GLOW_ACCENT, large=True),
            metric_card("推荐仓位", outlook.get("recommended_position","30%"), f"风险等级: {risk_level}", T.PURPLE, ""),
            metric_card("市场情绪", emotion.get("emotion","未知"),
                       f"涨跌比 {emotion.get('up_ratio',0):.0f}% | 涨{emotion.get('up_stocks',0)}跌{emotion.get('down_stocks',0)}",
                       T.GREEN if emotion.get("score",0.5) > 0.5 else T.RED),
            metric_card("涨跌中位数", f"{emotion.get('median_change',0):+.2f}%",
                       f"涨停{emotion.get('limit_up',0)} / 跌停{emotion.get('limit_down',0)}",
                       T.GREEN if emotion.get("median_change",0) > 0 else T.RED),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        # ── 第二行: 指数行情 + 操作建议 ──
        html.Div([
            # 指数行情
            html.Div([
                section_header("主要指数"),
                html.Div([
                    html.Div([
                        html.Div(row.get("name", ""), style={"fontSize":"13px","fontWeight":"600","color":T.TEXT}),
                        html.Div(f"{row.get('price',0):.2f}", style={"fontSize":"20px","fontWeight":"700","color":T.TEXT,"marginTop":"4px"}),
                        html.Div(f"{row.get('change_pct',0):+.2f}%", style={
                            "fontSize":"14px","fontWeight":"600","marginTop":"2px",
                            "color":T.GREEN if row.get("change_pct",0)>=0 else T.RED,
                        }),
                    ], style={"padding":"12px 16px","textAlign":"center","flex":"1",
                               "borderRight":f"1px solid {T.DIVIDER}"}) if not row.empty else html.Div()
                    for _, row in indices.iterrows()
                ], style={"display":"flex","background":"linear-gradient(135deg, #0D1525 0%, #0A1020 100%)","border":f"1px solid {T.BORDER}","borderRadius":"14px","overflow":"hidden"})
                if not indices.empty else html.Div("数据加载中..."),
            ], style={"flex":"2","marginRight":"16px"}),
            
            # 操作建议
            html.Div([
                section_header("操作指南"),
                html.Div([
                    html.Div(outlook.get("action_advice", ""), style={
                        "fontSize":"13px","color":T.TEXT,"lineHeight":"1.7",
                        "padding":"16px","background":"#0A0E17","borderRadius":"8px"
                    }),
                    html.Div([
                        html.Span("热点: ", style={"color":T.TEXT_SEC}),
                        html.Span("、".join(outlook.get("hot_sectors",["无"])[:3]) if outlook.get("hot_sectors") else "无",
                                 style={"color":T.GREEN}),
                    ], style={"marginTop":"12px","fontSize":"12px"}),
                    html.Div([
                        html.Span("回避: ", style={"color":T.TEXT_SEC}),
                        html.Span("、".join(outlook.get("cold_sectors",["无"])[:3]) if outlook.get("cold_sectors") else "无",
                                 style={"color":T.RED}),
                    ], style={"marginTop":"6px","fontSize":"12px"}),
                ]),
            ], style={"flex":"1", **{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])}}),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        # ── 第三行: 市场技术指标 ──
        html.Div([
            html.Div([
                section_header("技术面"),
                html.Table([
                    tr(k, v) for k, v in {
                        "短期趋势": overall.get("index_status",{}).get("short_trend","—"),
                        "中期趋势": overall.get("index_status",{}).get("mid_trend","—"),
                        "均线排列": overall.get("index_status",{}).get("ma_alignment","—"),
                        "近5日涨幅": f"{overall.get('index_status',{}).get('short_return',0):+.2f}%",
                        "近月涨幅": f"{overall.get('index_status',{}).get('monthly_return',0):+.2f}%",
                        "20日波动": f"{overall.get('index_status',{}).get('volatility_20d',0):.2f}%",
                    }.items()
                ]),
            ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])},"flex":"1","marginRight":"16px"}),
            
            html.Div([
                section_header("量能分析"),
                html.Table([
                    tr(k, v) for k, v in {
                        "成交量": f"{overall.get('volume_analysis',{}).get('current_volume',0):,}",
                        "20日均量": f"{overall.get('volume_analysis',{}).get('ma20_volume',0):,}",
                        "量能状态": overall.get("volume_analysis",{}).get("vol_status","正常"),
                        "涨跌比": f"{overall.get('breadth_analysis',{}).get('up_ratio',0):.0f}%",
                        "市场广度": overall.get("breadth_analysis",{}).get("breadth_status","均衡"),
                    }.items()
                ]),
            ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])},"flex":"1"}),
        ], style={"display":"flex"}),
    ])


def tr(k, v):
    return html.Tr([
        html.Td(k, style={"padding":"6px 0","fontSize":"13px","color":T.TEXT_SEC,"width":"40%"}),
        html.Td(v, style={"padding":"6px 0","fontSize":"13px","color":T.TEXT,"fontWeight":"600"}),
    ])


# ═════════════════════════════════════════════════════════════
#  投资决策页
# ═════════════════════════════════════════════════════════════

def render_decisions():
    engine = InvestmentDecisionEngine()
    picks = engine.top_picks(15)
    warnings = engine.risk_warnings(8)
    outlook = engine.market_outlook_report()
    
    perf = outlook.get("performance", {})
    
    return html.Div([
        # 绩效摘要条
        html.Div([
            metric_card("累计信号", str(perf.get("total_signals",0)), "次", T.ACCENT, GLOW_ACCENT),
            metric_card("历史胜率", f"{perf.get('buy_win_rate',0):.1f}%", "买入信号", T.GREEN if perf.get("buy_win_rate",0)>50 else T.AMBER, GLOW_GREEN if perf.get("buy_win_rate",0)>50 else ""),
            metric_card("均收益率", f"{perf.get('avg_return',0):+.2f}%", "每笔", T.GREEN if perf.get("avg_return",0)>0 else T.RED),
            metric_card("数据来源", outlook.get("data_freshness","实时"), "API", T.TEXT_SEC),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        # Top买入信号
        html.Div([
            section_header(f"📈 买入信号 ({len(picks)})"),
            html.Div([
                buy_signal_card(d) for d in picks
            ], style={"display":"grid","gridTemplateColumns":"repeat(auto-fill, minmax(380px, 1fr))","gap":"12px"} if picks else {}),
            html.Div("无买入信号 — 当前市场缺乏明确机会，建议观望", style={"color":T.TEXT_SEC,"padding":"40px","textAlign":"center"}) if not picks else None,
        ], style={"marginBottom":"24px"}),
        
        # 风险预警
        html.Div([
            section_header(f"⚠️ 风险预警 ({len(warnings)})"),
            html.Div([
                risk_card(d) for d in warnings
            ], style={"display":"grid","gridTemplateColumns":"repeat(auto-fill, minmax(320px, 1fr))","gap":"10px"} if warnings else {}),
            html.Div("✓ 当前无风险预警信号", style={"color":T.GREEN,"padding":"20px","textAlign":"center"}) if not warnings else None,
        ]),
        
        # 行业配置建议
        html.Div([
            section_header("🏭 行业配置建议"),
            html.Div([
                html.Div([
                    html.Div(outlook.get("portfolio_advice",{}).get("strategy",""), style={"fontSize":"15px","fontWeight":"700","color":T.ACCENT,"marginBottom":"12px"}),
                    *[html.Div([
                        html.Span(k, style={"fontSize":"13px","color":T.TEXT}),
                        html.Span(v, style={"fontSize":"13px","color":T.ACCENT,"fontWeight":"600","float":"right"}),
                    ], style={"padding":"6px 0"}) for k, v in outlook.get("portfolio_advice",{}).get("allocation",{}).items()],
                ], style={"padding":"16px"}),
            ], style={**{k:v for k,v in zip(["background","border","borderRadius"],[T.CARD,f"1px solid {T.BORDER}","10px"])}}),
        ], style={"marginTop":"20px","maxWidth":"380px"}),
    ])


# ═════════════════════════════════════════════════════════════
#  策略回测页
# ═════════════════════════════════════════════════════════════

def render_backtest():
    from hyperion.engine.backtest import BacktestEngine
    from hyperion.strategy.base import list_strategies
    
    engine = BacktestEngine()
    codes = ["600519", "000858", "300750", "601318", "000333", "002594", "600036", "002415"]
    
    rows = []
    for code in codes:
        df = engine.compare_strategies(code, days=180)
        if not df.empty:
            for _, row in df.iterrows():
                sharpe = row.get("sharpe", 0)
                if sharpe is None: sharpe = 0
                rows.append({
                    "标的": f"{get_stock_name(code)}",
                    "代码": code,
                    "策略": row["strategy"],
                    "评级": row["rating"],
                    "年化收益": row["annual_return"],
                    "夏普": f"{sharpe:.2f}" if isinstance(sharpe,(int,float)) else str(sharpe),
                    "最大回撤": row["max_drawdown"],
                    "胜率": row["win_rate"],
                    "交易次数": row["total_trades"],
                    "盈亏比": row["profit_factor"],
                })
    
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return html.Div("回测数据不足", style={"color":T.TEXT_SEC,"padding":"40px"})
    
    # 最好的策略
    best = df_all[df_all["评级"] != "N/A"].copy()
    
    return html.Div([
        html.Div([
            metric_card("回测策略", f"{len(list_strategies())} 个", "趋势/回归/动量/异动/多因子", T.PURPLE),
            metric_card("回测标的", f"{len(codes)} 只", "各行业龙头", T.ACCENT, GLOW_ACCENT),
            metric_card("回测区间", "180 天", "Sina真实数据", T.TEXT_SEC),
        ], style={"display":"flex","marginBottom":"24px"}),
        
        html.Div([
            section_header("📊 策略排名"),
            dash_table.DataTable(
                data=best.head(20).to_dict("records"),
                columns=[{"name":c,"id":c} for c in df_all.columns],
                style_header={"backgroundColor":T.CARD,"color":T.TEXT_SEC,"fontWeight":"600","border":f"1px solid {T.BORDER}"},
                style_cell={"backgroundColor":T.CARD,"color":T.TEXT,"border":f"1px solid {T.BORDER}","textAlign":"center","fontSize":"13px"},
                style_data_conditional=[
                    {"if":{"filter_query":"{评级} = 'A+' || {评级} = 'A'"},"color":T.GREEN,"fontWeight":"bold"},
                    {"if":{"filter_query":"{评级} = 'D'"},"color":T.RED},
                    {"if":{"filter_query":"{评级} = 'N/A'"},"color":T.TEXT_SEC},
                ],
                style_table={"overflowX":"auto","borderRadius":"10px"},
            ),
        ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])}}),
        
        html.Div([
            section_header("💡 策略解读"),
            html.Div([
                html.P("趋势跟踪 — 单边行情利器，震荡市需谨慎", style={"color":T.TEXT_SEC,"margin":"4px 0","fontSize":"13px"}),
                html.P("均值回归 — 震荡市优选，超卖买入等反弹", style={"color":T.TEXT_SEC,"margin":"4px 0","fontSize":"13px"}),
                html.P("动量突破 — 强势市场追涨，严格止损", style={"color":T.TEXT_SEC,"margin":"4px 0","fontSize":"13px"}),
                html.P("成交量异动 — 短线快进快出，持仓≤5天", style={"color":T.TEXT_SEC,"margin":"4px 0","fontSize":"13px"}),
                html.P("多因子Alpha — 综合评分，适合中长期配置", style={"color":T.TEXT_SEC,"margin":"4px 0","fontSize":"13px"}),
            ], style={"marginTop":"8px"}),
        ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])},"marginTop":"16px"}),
    ])


# ═════════════════════════════════════════════════════════════
#  绩效追踪页
# ═════════════════════════════════════════════════════════════

def render_performance():
    tracker = PerformanceTracker()
    summary = tracker.get_performance_summary()
    track_text = tracker.get_track_record_text()
    
    total = summary.get("total_signals_ever", 0)
    buy_total = summary.get("buy_signals_total", 0)
    buy_correct = summary.get("buy_signals_correct", 0)
    win_rate = summary.get("buy_win_rate", 0)
    avg_ret = summary.get("cumulative_avg_return", 0)
    strat_perf = summary.get("strategy_performance", {})
    
    wr_color = T.GREEN if win_rate > 55 else (T.AMBER if win_rate > 40 else T.RED)
    
    return html.Div([
        html.Div([
            metric_card("累计信号", str(total), f"已验证 {summary.get('verified_signals_ever',0)}", T.ACCENT, GLOW_ACCENT, large=True),
            metric_card("买入信号", str(buy_total), f"盈利 {buy_correct} 次", T.TEXT),
            metric_card("胜率", f"{win_rate:.1f}%", "买入方向", wr_color, GLOW_GREEN if win_rate>50 else GLOW_RED),
            metric_card("均收益率", f"{avg_ret:+.2f}%", "每笔买入建议", T.GREEN if avg_ret > 0 else T.RED),
        ], style={"display":"flex","marginBottom":"24px"}),
        
        # 策略绩效排名
        html.Div([
            section_header("🏆 策略绩效排名"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(name, style={"fontSize":"14px","fontWeight":"600","color":T.TEXT}),
                        html.Span(f"  {perf.get('trades',0)}笔", style={"fontSize":"12px","color":T.TEXT_SEC}),
                    ]),
                    html.Div([
                        html.Span(f"胜率 {perf.get('win_rate',0):.1f}%", style={"fontSize":"13px","color":T.GREEN if perf.get('win_rate',0)>50 else T.RED,"fontWeight":"600","marginRight":"16px"}),
                        html.Span(f"均收益 {perf.get('avg_return',0):+.2f}%", style={"fontSize":"13px","color":T.GREEN if perf.get('avg_return',0)>0 else T.RED}),
                    ], style={"marginTop":"4px"}),
                    # Progress bar
                    html.Div([
                        html.Div(style={
                            "width":f"{min(100, perf.get('win_rate',0))}%",
                            "height":"3px","borderRadius":"2px",
                            "background":T.GREEN if perf.get('win_rate',0)>50 else T.RED,
                        })
                    ], style={"background":T.BORDER,"borderRadius":"2px","marginTop":"6px"}),
                ], style={"padding":"12px 16px","borderBottom":f"1px solid {T.DIVIDER}"})
                for name, perf in sorted(strat_perf.items(), key=lambda x: x[1].get("win_rate",0), reverse=True)
            ], style={**{k:v for k,v in zip(["background","border","borderRadius"],[T.CARD,f"1px solid {T.BORDER}","10px"])}}),
        ], style={"flex":"1","marginRight":"20px"}),
        
        # 最近信号
        html.Div([
            section_header("📝 最近验证信号"),
            html.Div([
                html.Div([
                    html.Span("✅" if s.get("realized_return",0)>0 else "❌", style={"fontSize":"18px","marginRight":"8px"}),
                    html.Span(f"{s.get('name','')}({s.get('code','')})", style={"fontSize":"13px","color":T.TEXT,"fontWeight":"600"}),
                    html.Span(f" {s.get('realized_return',0):+.2f}%", style={
                        "fontSize":"13px","fontWeight":"600",
                        "color":T.GREEN if s.get("realized_return",0)>0 else T.RED,
                        "float":"right"
                    }),
                ], style={"padding":"8px 0","borderBottom":f"1px solid {T.DIVIDER}"})
                for s in summary.get("recent_signals", [])[-8:]
            ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","16px"])}}) if summary.get("recent_signals") else html.Div("暂无验证数据", style={"color":T.TEXT_SEC,"padding":"20px"}),
        ], style={"flex":"1"}),
    ], style={"display":"flex"})


# ═════════════════════════════════════════════════════════════
#  风险管理页
# ═════════════════════════════════════════════════════════════

def render_risk():
    try:
        engine = InvestmentDecisionEngine()
        picks = engine.top_picks(20)
        holdings = {}
        for i, d in enumerate(picks[:8]):
            holdings[d.code] = d.max_position_pct / 100.0
        total = sum(holdings.values())
        if total > 0:
            holdings = {k: v/total for k, v in holdings.items()}
        
        rm = RiskManager()
        report = rm.assess_portfolio(holdings)
    except Exception as e:
        return html.Div(f"风险模型加载中... {e}", style={"color":T.TEXT_SEC,"padding":"40px"})
    
    risk_color = {"low":T.GREEN,"medium":T.AMBER,"high":T.RED,"critical":T.RED}.get(report.risk_level,T.TEXT_SEC)
    
    return html.Div([
        html.Div([
            metric_card("风险等级", report.risk_level.upper(), report.advice[:60]+"...", risk_color, GLOW_RED if report.risk_level in ("high","critical") else "", large=True),
            metric_card("VaR 95%", f"{report.var_95_daily_pct*100:.2f}%", f"99%: {report.var_99_daily*100:.2f}% | CVaR: {report.cvar_95_daily*100:.2f}%", T.TEXT),
            metric_card("最大回撤", f"{report.max_drawdown*100:.2f}%",
                       f"当前: {report.current_drawdown*100:.2f}% | 持续{report.max_drawdown_duration}天",
                       T.RED if report.max_drawdown < -0.20 else T.AMBER),
            metric_card("分散化", f"有效头寸 {report.effective_n:.1f}",
                       f"HHI: {report.concentration_risk:.3f} | Beta: {report.beta_to_market:.2f}",
                       T.GREEN if report.effective_n > 5 else T.AMBER),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        html.Div([
            section_header("⚡ 压力测试"),
            html.Div([
                html.Div([
                    html.Span(name, style={"fontSize":"13px","color":T.TEXT}),
                    html.Div(style={"width":"100%","height":"6px","background":T.BORDER,"borderRadius":"3px","marginTop":"8px"}),
                    html.Div(style={
                        "width":f"{min(100, abs(loss)*100)}%",
                        "height":"6px","marginTop":"-6px","background":T.RED,"borderRadius":"3px",
                    }),
                    html.Span(f"{loss*100:.0f}%", style={"fontSize":"11px","color":T.RED,"float":"right","marginTop":"-14px"}),
                ], style={"flex":"1","margin":"0 10px"})
                for name, loss in report.stress_tests.items()
            ], style={"display":"flex","marginTop":"12px"}),
        ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])},"marginBottom":"20px"}),
        
        html.Div([
            section_header("🚨 风险警告"),
            *([html.Div([
                html.Span("⚡ ", style={"fontSize":"14px"}),
                html.Span(w, style={"fontSize":"13px","color":T.AMBER}),
            ], style={"padding":"8px 12px","marginBottom":"6px","background":f"{T.AMBER}10","borderRadius":"6px"}) for w in report.warnings]
            if report.warnings else [html.Div("✓ 未检测到显著风险", style={"color":T.GREEN,"padding":"12px"})]),
        ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])}}),
    ])


# ═════════════════════════════════════════════════════════════
#  报告中心页
# ═════════════════════════════════════════════════════════════

def render_reports():
    reports = ReportGenerator.list_reports()
    
    return html.Div([
        html.Div([
            section_header("📄 已生成报告"),
            *([html.Div([
                html.Span(r["filename"], style={"fontSize":"14px","color":T.TEXT,"fontWeight":"600"}),
                html.Span(f"  {r['size_kb']:.1f}KB", style={"fontSize":"12px","color":T.TEXT_SEC}),
            ], style={"padding":"10px 0","borderBottom":f"1px solid {T.DIVIDER}"}) for r in reports]
            if reports else [html.Div("暂无报告，使用 python hyperion/run.py --brief 生成", style={"color":T.TEXT_SEC,"padding":"20px"})]),
        ], style={**{k:v for k,v in zip(["background","border","borderRadius","padding"],[T.CARD,f"1px solid {T.BORDER}","10px","20px"])}}),
    ])


# ═════════════════════════════════════════════════════════════
#  Server
# ═════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════
#  信号预警页
# ═════════════════════════════════════════════════════════════

def render_alerts():
    alerts_system = SignalAlertSystem()
    alerts = alerts_system.scan_all()
    
    criticals = [a for a in alerts if a.level == AlertLevel.CRITICAL]
    warnings = [a for a in alerts if a.level == AlertLevel.WARNING]
    infos = [a for a in alerts if a.level == AlertLevel.INFO]
    
    def alert_card(a, level_color, icon):
        cat_labels = {"price":"价格","technical":"技术","volume":"量能","sector":"板块","sentiment":"情绪"}
        cat = cat_labels.get(a.category, a.category)
        expiry_dt = datetime.fromisoformat(a.expiry) if a.expiry else None
        expiry_str = expiry_dt.strftime("%H:%M") if expiry_dt and expiry_dt.date() == datetime.now().date() else (a.expiry[:16] if a.expiry else "")
        return html.Div([
            html.Div([
                html.Span(icon, style={"fontSize":"20px","marginRight":"10px"}),
                html.Div([
                    html.Div([
                        html.Span(a.title, style={"fontSize":"14px","fontWeight":"700","color":T.TEXT}),
                        html.Span(f"  {cat}", style={"fontSize":"11px","color":T.TEXT_SEC,"padding":"1px 8px","background":T.CARD_ALT,"borderRadius":"8px"}),
                    ]),
                    html.Div(a.description, style={"fontSize":"12px","color":T.TEXT_SEC,"marginTop":"4px","lineHeight":"1.5"}),
                ], style={"flex":"1"}),
            ], style={"display":"flex","alignItems":"flex-start"}),
            html.Div([
                html.Div(a.action_advice, style={
                    "fontSize":"12px","color":T.ACCENT,"marginTop":"8px","padding":"8px 12px",
                    "background":f"{T.ACCENT}08","borderRadius":"6px","border":f"1px solid {T.ACCENT}15","lineHeight":"1.5"
                }) if a.action_advice else None,
                html.Div([
                    html.Span(f"触发值: {a.current_value:.2f}", style={"color":T.TEXT_SEC,"fontSize":"11px","marginRight":"16px"}),
                    html.Span(f"阈值: {a.threshold:.2f}", style={"color":T.TEXT_SEC,"fontSize":"11px"}) if a.threshold else None,
                    html.Span(f" 有效期至: {expiry_str}", style={"color":T.AMBER,"fontSize":"10px","float":"right"}) if expiry_str else None,
                ], style={"marginTop":"8px"}),
            ]),
        ], style={
            "background":f"linear-gradient(135deg, {T.CARD} 0%, {T.CARD_ALT} 100%)",
            "border":f"1px solid {level_color}30",
            "borderRadius":"10px",
            "padding":"16px",
            "borderLeft":f"4px solid {level_color}",
            "marginBottom":"8px"
        })
    
    return html.Div([
        # Summary bar
        html.Div([
            metric_card("🚨 紧急", str(len(criticals)), "条", T.RED, GLOW_RED),
            metric_card("⚡ 警告", str(len(warnings)), "条", T.AMBER, GLOW_RED),
            metric_card("ℹ️ 提示", str(len(infos)), "条", T.ACCENT, GLOW_ACCENT),
            metric_card("数据", datetime.now().strftime("%H:%M"), "实时扫描", T.TEXT_SEC),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        # Critical alerts
        html.Div([
            section_header(f"🚨 紧急预警 ({len(criticals)})"),
            html.Div([alert_card(a, T.RED, "🔴") for a in criticals],
                     style={"maxHeight":"600px","overflowY":"auto"}) if criticals else
            html.Div("✅ 当前无紧急预警", style={"color":T.GREEN,"padding":"16px","textAlign":"center"}),
        ], style={"marginBottom":"16px"}) if criticals else None,
        
        # Warning alerts
        html.Div([
            section_header(f"⚡ 预警信号 ({len(warnings)})"),
            html.Div([alert_card(a, T.AMBER, "🟡") for a in warnings],
                     style={"maxHeight":"600px","overflowY":"auto"}) if warnings else
            html.Div("✅ 当前无预警信号", style={"color":T.GREEN,"padding":"16px","textAlign":"center"}),
        ], style={"marginBottom":"16px"}) if warnings else None,
        
        # Info alerts
        html.Div([
            section_header(f"ℹ️ 市场提示 ({len(infos)})"),
            html.Div([alert_card(a, T.ACCENT, "🔵") for a in infos],
                     style={"maxHeight":"400px","overflowY":"auto"}) if infos else None,
        ]) if infos else None,
        
        # Empty state
        html.Div("✅ 当前无活跃预警信号", style={"color":T.GREEN,"padding":"40px","textAlign":"center","fontSize":"16px"}) if not alerts else None,
    ])


# ═════════════════════════════════════════════════════════════
#  交易日志页
# ═════════════════════════════════════════════════════════════

def render_journal():
    journal = TradeJournal()
    overview = journal.performance_overview()
    open_trades = journal.get_open_trades()
    closed_trades = journal.get_closed_trades()
    
    wr_color = T.GREEN if overview.get('win_rate', 0) > 50 else T.AMBER
    
    # Format trade rows
    def trade_row(t, is_open=True):
        emoji = "📈" if is_open else ("✅" if t.get('pnl', 0) > 0 else "❌")
        pnl_color = T.GREEN if t.get('pnl', 0) > 0 else T.RED
        return html.Tr([
            html.Td(emoji, style={"padding":"8px 4px","fontSize":"14px"}),
            html.Td(t.get("name",""), style={"padding":"8px 4px","fontSize":"13px","fontWeight":"600","color":T.TEXT}),
            html.Td(t.get("code",""), style={"padding":"8px 4px","fontSize":"12px","color":T.TEXT_SEC}),
            html.Td(f"¥{t.get('entry_price',0):.2f}", style={"padding":"8px 4px","fontSize":"12px","color":T.TEXT}),
            html.Td(f"¥{t.get('exit_price',0):.2f}" if not is_open else "—", style={"padding":"8px 4px","fontSize":"12px","color":T.TEXT}),
            html.Td(f"{t.get('pnl_pct',0):+.2f}%" if not is_open else "—", style={"padding":"8px 4px","fontSize":"12px","fontWeight":"700","color":pnl_color}),
            html.Td(t.get("exit_reason","持仓中"), style={"padding":"8px 4px","fontSize":"11px","color":T.TEXT_SEC}),
            html.Td(t.get("holding_days",0), style={"padding":"8px 4px","fontSize":"12px","color":T.TEXT_SEC}),
        ])
    
    return html.Div([
        # Summary bar
        html.Div([
            metric_card("总交易", str(overview.get('total_trades',0)), f"已平仓{overview.get('closed_trades',0)}", T.ACCENT, GLOW_ACCENT),
            metric_card("胜率", f"{overview.get('win_rate',0):.1f}%", f"盈利{overview.get('win_trades',0)}/亏损{overview.get('loss_trades',0)}", wr_color),
            metric_card("总盈亏", f"¥{overview.get('total_pnl',0):+,.2f}", f"均盈亏¥{overview.get('avg_pnl_per_trade',0):+,.2f}", T.GREEN if overview.get('total_pnl',0)>0 else T.RED),
            metric_card("盈亏比", f"{overview.get('profit_factor',0):.2f}", f"均盈利{overview.get('avg_win_pct',0):+.2f}/均亏损{overview.get('avg_loss_pct',0):.2f}", T.ACCENT),
        ], style={"display":"flex","marginBottom":"20px"}),
        
        # System signal tracking
        html.Div([
            section_header("📊 系统信号跟踪"),
            html.Div([
                html.Div([
                    html.Div("跟随系统信号", style={"fontSize":"13px","color":T.TEXT,"fontWeight":"600"}),
                    html.Div(f"胜率 {overview.get('followed_signal_win_rate',0):.1f}%", style={"fontSize":"24px","fontWeight":"800","color":T.GREEN if overview.get('followed_signal_win_rate',0) > 50 else T.RED}),
                ], style={"flex":"1","padding":"12px","borderRight":f"1px solid {T.DIVIDER}"}),
                html.Div([
                    html.Div("自主决策", style={"fontSize":"13px","color":T.TEXT,"fontWeight":"600"}),
                    html.Div(f"胜率 {overview.get('unfollowed_signal_win_rate',0):.1f}%", style={"fontSize":"24px","fontWeight":"800","color":T.GREEN if overview.get('unfollowed_signal_win_rate',0) > 50 else T.RED}),
                ], style={"flex":"1","padding":"12px"}),
                html.Div([
                    html.Div("💡", style={"fontSize":"24px","marginBottom":"4px"}),
                    html.Div(
                        "建议信任系统信号" if overview.get('followed_signal_win_rate',0) >= overview.get('unfollowed_signal_win_rate',0) else "建议优化系统或结合自主判断",
                        style={"fontSize":"12px","color":T.ACCENT,"fontWeight":"600"}
                    ),
                ], style={"flex":"1","padding":"12px","background":f"{T.ACCENT}08","borderRadius":"8px"}),
            ], style={"display":"flex","background":T.CARD,"border":f"1px solid {T.BORDER}","borderRadius":"10px"}),
        ], style={"marginBottom":"16px"}),
        
        # Open positions
        html.Div([
            section_header(f"📌 当前持仓 ({len(open_trades)})"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("名称", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("代码", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("入场价", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("出场价", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("盈亏", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("状态", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    html.Th("天数", style={"padding":"8px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                ])),
                html.Tbody([trade_row(t, True) for t in open_trades] if open_trades else [
                    html.Tr(html.Td("暂无持仓", colSpan=8, style={"padding":"24px","textAlign":"center","color":T.TEXT_SEC}))
                ]),
            ], style={"width":"100%","borderCollapse":"collapse","background":T.CARD,
                       "border":f"1px solid {T.BORDER}","borderRadius":"10px","overflow":"hidden"})
            if True else html.Div()
        ], style={"marginBottom":"16px"}),
        
        # Closed trades
        html.Div([
            section_header(f"📋 历史平仓 ({len(closed_trades)})"),
            html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("名称", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("代码", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("入场", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("出场", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("盈亏", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("理由", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                        html.Th("持有", style={"padding":"6px 4px","color":T.TEXT_SEC,"fontSize":"11px"}),
                    ])),
                    html.Tbody([trade_row(t, False) for t in closed_trades[-10:]][::-1] if closed_trades else [
                        html.Tr(html.Td("暂无平仓记录", colSpan=8, style={"padding":"24px","textAlign":"center","color":T.TEXT_SEC}))
                    ]),
                ], style={"width":"100%","borderCollapse":"collapse","background":T.CARD,
                           "border":f"1px solid {T.BORDER}","borderRadius":"10px","overflow":"hidden"})
            ], style={"maxHeight":"400px","overflowY":"auto"}),
        ]),
    ])


def run_server(host="127.0.0.1", port=8050, debug=False):
    app = create_app()
    if app is None:
        return
    print(f"\n  Hyperion Pro 仪表盘")
    print(f"  → http://{host}:{port}")
    print()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
