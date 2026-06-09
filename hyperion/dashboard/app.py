"""
Hyperion Pro — Web 仪表盘
===========================
基于 Plotly Dash 构建的交互式量化交易仪表盘

功能模块:
  1. 市场总览 — 指数行情 + 大盘判断 + 情绪指标
  2. 个股分析 — 技术指标图表 + 资金流向 + 信号
  3. 行业轮动 — 行业排行 + 推荐配置
  4. 股票扫描 — Top买入/卖出信号 + 筛选
  5. 组合管理 — 持仓分析 + 诊断建议
  6. 投资报告 — 查看已生成的报告

使用: python hyperion/run.py --server
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 确保项目路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from hyperion.data.market import (
    fetch_realtime_quotes, fetch_index_quotes, fetch_history,
    sector_quotes, scan_market, money_flow,
    CORE_STOCKS, INDICES, get_stock_name, get_stock_industry
)
from hyperion.analysis.market_state import MarketStateAnalyzer
from hyperion.analysis.technical import TechnicalAnalyzer
from hyperion.analysis.signals import SignalGenerator
from hyperion.analysis.decision_engine import InvestmentDecisionEngine
from hyperion.risk import RiskManager, quick_risk_check
from hyperion.reporting.report_generator import ReportGenerator, REPORTS_DIR

# ==========================================================
#  尝试导入 Dash
# ==========================================================
try:
    import dash
    from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

# ==========================================================
#  CSS 样式常量 — 深色专业量化面板风格
# ==========================================================
COLORS = {
    "bg": "#0d1117",
    "card_bg": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_secondary": "#8b949e",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "yellow": "#d29922",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
}

CARD_STYLE = {
    "background": COLORS["card_bg"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "16px",
    "margin": "8px 0",
}

HEADER_STYLE = {
    "color": COLORS["accent"],
    "fontSize": "14px",
    "fontWeight": "600",
    "marginBottom": "8px",
    "textTransform": "uppercase",
    "letterSpacing": "1px",
}


def create_app(debug: bool = False):
    """创建 Dash 应用"""
    if not HAS_DASH:
        print("请安装: pip install dash plotly")
        return None
    
    app = dash.Dash(
        __name__,
        title="Hyperion Pro 量化交易系统",
        assets_ignore=".*",
        suppress_callback_exceptions=True,
    )
    
    # 自定义 CSS
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    background-color: #0d1117;
                    color: #e6edf3;
                    margin: 0;
                    padding: 0;
                }
                * {
                    box-sizing: border-box;
                }
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                ::-webkit-scrollbar-track {
                    background: #0d1117;
                }
                ::-webkit-scrollbar-thumb {
                    background: #30363d;
                    border-radius: 4px;
                }
                .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
                    --accent: #58a6ff;
                    --border: #30363d;
                    --text-color: #e6edf3;
                    --selected-background: #1f2937;
                }
                .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
                    color: #e6edf3;
                    background-color: #161b22;
                    border: 1px solid #30363d;
                }
                .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                    background-color: #1c2128;
                    color: #8b949e;
                    border: 1px solid #30363d;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>{%config%}{%scripts%}{%renderer%}</footer>
        </body>
    </html>
    '''
    
    # ==========================================================
    #  布局
    # ==========================================================
    
    app.layout = html.Div([
        # 隐藏的数据存储
        dcc.Interval(id="refresh-interval", interval=300000),  # 5分钟刷新
        dcc.Store(id="signal-data"),
        
        # 顶部导航
        html.Div([
            html.Div([
                html.Span("HYPERION PRO", style={
                    "fontSize": "24px", "fontWeight": "700",
                    "color": COLORS["accent"], "letterSpacing": "2px"
                }),
                html.Span(" 量化交易系统", style={
                    "fontSize": "14px", "color": COLORS["text_secondary"],
                    "marginLeft": "12px"
                }),
            ], style={"padding": "16px 24px", "display": "inline-block"}),
            
            html.Div([
                html.Span(id="current-time", style={
                    "color": COLORS["text_secondary"], "fontSize": "12px",
                    "marginRight": "24px"
                }),
            ], style={"float": "right", "padding": "22px 24px"}),
        ], style={
            "background": COLORS["card_bg"],
            "borderBottom": f"1px solid {COLORS['border']}",
        }),
        
        # Tab 导航
        html.Div([
            dcc.Tabs(id="tabs", value="tab-overview", children=[
                dcc.Tab(label="市场总览", value="tab-overview", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="个股分析", value="tab-stock", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="行业轮动", value="tab-sectors", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="股票扫描", value="tab-scan", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="投资决策", value="tab-decisions", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="策略回测", value="tab-backtest", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="风险管控", value="tab-risk", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
                dcc.Tab(label="投资报告", value="tab-reports", style={
                    "background": COLORS["card_bg"], "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "padding": "8px 20px", "fontWeight": "500",
                }, selected_style={
                    "background": COLORS["card_bg"], "color": COLORS["accent"],
                    "borderTop": f"2px solid {COLORS['accent']}",
                    "borderBottom": "none",
                    "fontWeight": "600",
                }),
            ], style={
                "background": COLORS["bg"],
                "border": "none",
            }),
        ]),
        
        # Tab 内容
        html.Div(id="tab-content", style={"padding": "20px 24px"}),
    ])
    
    # ==========================================================
    #  Callbacks
    # ==========================================================
    
    @app.callback(
        Output("current-time", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_time(n):
        return f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
    )
    def render_tab(tab_value):
        if tab_value == "tab-overview":
            return _render_overview()
        elif tab_value == "tab-stock":
            return _render_stock_analysis()
        elif tab_value == "tab-sectors":
            return _render_sectors()
        elif tab_value == "tab-scan":
            return _render_scan()
        elif tab_value == "tab-decisions":
            return _render_decisions()
        elif tab_value == "tab-risk":
            return _render_risk()
        elif tab_value == "tab-reports":
            return _render_reports()
        return html.Div()
    
    # 个股分析回调
    @app.callback(
        [Output("stock-chart", "figure"),
         Output("stock-signal", "children"),
         Output("stock-moneyflow", "children")],
        [Input("stock-selector", "value"),
         Input("chart-period", "value")],
    )
    def update_stock_analysis(code, days):
        return _update_stock_chart(code, days)
    
    # 报告查看回调
    @app.callback(
        Output("report-content", "children"),
        Input("report-selector", "value"),
    )
    def view_report(report_path):
        if not report_path:
            return html.Div("选择报告查看", style={"color": COLORS["text_secondary"]})
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            return html.Pre(content, style={
                "background": COLORS["bg"],
                "color": COLORS["text"],
                "padding": "16px",
                "borderRadius": "8px",
                "border": f"1px solid {COLORS['border']}",
                "overflow": "auto",
                "maxHeight": "600px",
                "fontSize": "13px",
                "lineHeight": "1.6",
                "whiteSpace": "pre-wrap",
            })
        except Exception:
            return html.Div("读取报告失败")
    
    return app


# ==========================================================
#  页面渲染函数
# ==========================================================

def _render_overview():
    """市场总览页面"""
    overall = MarketStateAnalyzer.analyze_overall()
    emotion = MarketStateAnalyzer.analyze_emotion()
    indices = fetch_index_quotes()
    quotes = fetch_realtime_quotes()
    
    market_state = overall.get("market_state", "未知")
    state_colors = {
        "牛市": COLORS["green"], "反弹": COLORS["cyan"],
        "震荡": COLORS["yellow"], "回调": COLORS["red"],
        "熊市": COLORS["red"], "不明朗": COLORS["text_secondary"],
    }
    state_color = state_colors.get(market_state, COLORS["text"])
    
    children = [
        # 市场状态仪表卡
        html.Div([
            html.Div([
                html.Span("市场状态", style=HEADER_STYLE),
                html.Div([
                    html.Span(market_state, style={
                        "fontSize": "36px", "fontWeight": "700",
                        "color": state_color,
                    }),
                    html.Span(f"  {overall.get('confidence', 0)*100:.0f}% 置信度", style={
                        "fontSize": "14px", "color": COLORS["text_secondary"],
                        "marginLeft": "12px",
                    }),
                ]),
                html.Div(overall.get("advice", ""), style={
                    "fontSize": "14px", "color": COLORS["text"], "marginTop": "8px",
                }),
            ], style={"flex": "1", **CARD_STYLE, "margin": "0"}),
            
            html.Div([
                html.Span("推荐仓位", style=HEADER_STYLE),
                html.Div(overall.get("recommended_position", "30%"), style={
                    "fontSize": "28px", "fontWeight": "700", "color": COLORS["purple"],
                }),
                html.Div(f"风险等级: {overall.get('risk_level', '中')}", style={
                    "fontSize": "13px", "color": COLORS["text_secondary"], "marginTop": "4px",
                }),
            ], style={"flex": "1", **CARD_STYLE, "margin": "0 0 0 12px"}),
            
            html.Div([
                html.Span("市场情绪", style=HEADER_STYLE),
                html.Div(emotion.get("emotion", "未知"), style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": COLORS["green"] if emotion.get("score", 0.5) > 0.6 else COLORS["red"] if emotion.get("score", 0.5) < 0.4 else COLORS["yellow"],
                }),
                html.Div([
                    f"涨跌比 {emotion.get('up_ratio', 0):.0f}% | "
                    f"涨停 {emotion.get('limit_up', 0)} | "
                    f"跌停 {emotion.get('limit_down', 0)}"
                ], style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginTop": "4px"}),
            ], style={"flex": "1", **CARD_STYLE, "margin": "0 0 0 12px"}),
            
            html.Div([
                html.Span("涨跌幅中位数", style=HEADER_STYLE),
                html.Div(f"{emotion.get('median_change', 0):+.2f}%", style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": COLORS["green"] if emotion.get("median_change", 0) > 0 else COLORS["red"],
                }),
                html.Div(f"上涨 {emotion.get('up_stocks', 0)} / 下跌 {emotion.get('down_stocks', 0)}", style={
                    "fontSize": "12px", "color": COLORS["text_secondary"], "marginTop": "4px",
                }),
            ], style={"flex": "1", **CARD_STYLE, "margin": "0 0 0 12px"}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        
        # 指数行情
        html.Div([
            html.Span("主要指数行情", style=HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Div(index.get("name", ""), style={
                        "fontSize": "14px", "fontWeight": "600",
                    }),
                    html.Div(f"{index.get('price', 0):.2f}", style={
                        "fontSize": "22px", "fontWeight": "700", "color": COLORS["text"],
                        "marginTop": "4px",
                    }),
                    html.Div(f"{index.get('change_pct', 0):+.2f}%", style={
                        "fontSize": "16px", "fontWeight": "500",
                        "color": COLORS["green"] if index.get("change_pct", 0) >= 0 else COLORS["red"],
                    }),
                ], style={
                    "background": COLORS["card_bg"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "8px",
                    "padding": "12px 16px",
                    "flex": "1",
                    "textAlign": "center",
                }) for _, index in indices.iterrows()
            ]) if not indices.empty else [html.Div("数据加载中...")],
        ], style={"display": "flex", "gap": "8px", "marginBottom": "16px"}),
        
        # 技术指标详情
        html.Div([
            html.Div([
                html.Span("技术指标", style=HEADER_STYLE),
                html.Table([
                    html.Tr([html.Td("短期趋势", style={"padding": "4px 12px"}), 
                             html.Td(f"{overall.get('index_status', {}).get('short_trend', '未知')}", style={"padding": "4px 12px", "fontWeight": "600"})]),
                    html.Tr([html.Td("中期趋势"), html.Td(f"{overall.get('index_status', {}).get('mid_trend', '未知')}", style={"fontWeight": "600"})]),
                    html.Tr([html.Td("均线排列"), html.Td(f"{overall.get('index_status', {}).get('ma_alignment', '未知')}", style={"fontWeight": "600"})]),
                    html.Tr([html.Td("近5日涨幅"), html.Td(f"{overall.get('index_status', {}).get('short_return', 0):+.2f}%")]),
                    html.Tr([html.Td("近月涨幅"), html.Td(f"{overall.get('index_status', {}).get('monthly_return', 0):+.2f}%")]),
                    html.Tr([html.Td("20日波动率"), html.Td(f"{overall.get('index_status', {}).get('volatility_20d', 0):.2f}%")]),
                    html.Tr([html.Td("量能状态"), html.Td(f"{overall.get('volume_analysis', {}).get('vol_status', '正常')}", style={"fontWeight": "600"})]),
                ], style={"color": COLORS["text"], "fontSize": "13px"})
            ], style={**CARD_STYLE, "flex": "1"}),
            
            html.Div([
                html.Span("涨跌分布", style=HEADER_STYLE),
                html.Div([
                    html.Div([
                        html.Div(f"{overall.get('breadth_analysis', {}).get('up_stocks', 0)}", 
                                 style={"fontSize": "24px", "fontWeight": "700", "color": COLORS["green"]}),
                        html.Div("上涨", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                    ], style={"textAlign": "center", "flex": "1"}),
                    html.Div([
                        html.Div(f"{overall.get('breadth_analysis', {}).get('down_stocks', 0)}",
                                 style={"fontSize": "24px", "fontWeight": "700", "color": COLORS["red"]}),
                        html.Div("下跌", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                    ], style={"textAlign": "center", "flex": "1"}),
                    html.Div([
                        html.Div(f"{emotion.get('limit_up', 0)}",
                                 style={"fontSize": "24px", "fontWeight": "700", "color": COLORS["yellow"]}),
                        html.Div("涨停", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                    ], style={"textAlign": "center", "flex": "1"}),
                    html.Div([
                        html.Div(f"{emotion.get('limit_down', 0)}",
                                 style={"fontSize": "24px", "fontWeight": "700", "color": COLORS["red"]}),
                        html.Div("跌停", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                    ], style={"textAlign": "center", "flex": "1"}),
                ], style={"display": "flex", "marginTop": "12px"}),
                
                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),
                
                html.Span("操作策略", style=HEADER_STYLE),
                html.Div(overall.get("advice", ""), style={
                    "fontSize": "14px", "color": COLORS["text"], "marginTop": "8px",
                    "lineHeight": "1.6",
                }),
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "gap": "12px"}),
    ]
    
    return html.Div(children)


def _render_stock_analysis():
    """个股分析页面"""
    default_code = "600519"
    
    return html.Div([
        html.Div([
            html.Div([
                html.Span("选择股票", style=HEADER_STYLE),
                dcc.Dropdown(
                    id="stock-selector",
                    options=[
                        {"label": f"{name} ({code})", "value": code}
                        for code, name, _ in CORE_STOCKS
                    ],
                    value=default_code,
                    style={
                        "background": COLORS["bg"],
                        "color": COLORS["text"],
                        "border": f"1px solid {COLORS['border']}",
                    },
                    clearable=False,
                ),
            ], style={"flex": "1", "marginRight": "12px"}),
            
            html.Div([
                html.Span("分析周期", style=HEADER_STYLE),
                dcc.Dropdown(
                    id="chart-period",
                    options=[
                        {"label": "60天", "value": 60},
                        {"label": "120天", "value": 120},
                        {"label": "250天", "value": 250},
                        {"label": "500天", "value": 500},
                    ],
                    value=120,
                    style={
                        "background": COLORS["bg"],
                        "color": COLORS["text"],
                        "border": f"1px solid {COLORS['border']}",
                    },
                    clearable=False,
                ),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        
        # K线图
        html.Div([
            dcc.Graph(id="stock-chart", style={"height": "500px"}),
        ], style=CARD_STYLE),
        
        html.Div([
            # 信号
            html.Div(id="stock-signal", style={**CARD_STYLE, "flex": "1"}),
            # 资金流向
            html.Div(id="stock-moneyflow", style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ])


def _update_stock_chart(code, days):
    """更新个股分析图表"""
    history = fetch_history(code, days=days)
    
    if history.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLORS["card_bg"],
            plot_bgcolor=COLORS["card_bg"],
            annotations=[{"text": "数据不足", "showarrow": False, "font": {"color": COLORS["text_secondary"], "size": 20}}],
        )
        return empty_fig, html.Div("数据不足"), html.Div("数据不足")
    
    # 技术分析
    analyzer = TechnicalAnalyzer()
    tech = analyzer.comprehensive_analysis(history)
    sig_gen = SignalGenerator()
    signal = sig_gen.analyze_stock(code)
    
    # 创建图表
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=("价格 K线", "成交量", "MACD"),
    )
    
    close = history["close"].values
    dates = history.index if "date" not in history.columns else history["date"]
    
    # K线
    fig.add_trace(go.Candlestick(
        x=dates,
        open=history["open"].values,
        high=history["high"].values,
        low=history["low"].values,
        close=close,
        name=code,
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
    ), row=1, col=1)
    
    # 均线
    ma5 = pd.Series(close).rolling(5).mean()
    ma20 = pd.Series(close).rolling(20).mean()
    ma60 = pd.Series(close).rolling(60).mean()
    
    fig.add_trace(go.Scatter(x=dates, y=ma5, name="MA5", line={"color": "#f0b429", "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma20, name="MA20", line={"color": "#58a6ff", "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ma60, name="MA60", line={"color": "#bc8cff", "width": 1}), row=1, col=1)
    
    # 布林带
    ma = pd.Series(close).rolling(20).mean()
    std = pd.Series(close).rolling(20).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    
    fig.add_trace(go.Scatter(x=dates, y=upper, name="上轨", 
                             line={"color": "rgba(88,166,255,0.3)", "width": 1}, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lower, name="下轨",
                             line={"color": "rgba(88,166,255,0.3)", "width": 1}, 
                             fill="tonexty", fillcolor="rgba(88,166,255,0.05)", showlegend=False), row=1, col=1)
    
    # 成交量
    colors = [COLORS["green"] if c >= o else COLORS["red"] 
              for c, o in zip(close, history["open"].values)]
    
    fig.add_trace(go.Bar(x=dates, y=history["volume"].values, name="量",
                          marker_color=colors, opacity=0.7), row=2, col=1)
    
    # MACD
    macd = tech.get("macd", {})
    fig.add_trace(go.Bar(x=dates, y=[0]*len(dates), name="柱", showlegend=False), row=3, col=1)
    
    # 用简单计算
    ema12 = pd.Series(close).ewm(span=12).mean()
    ema26 = pd.Series(close).ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_val = 2 * (dif - dea)
    
    macd_colors = [COLORS["red"] if v < 0 else COLORS["green"] for v in macd_val]
    fig.add_trace(go.Bar(x=dates, y=macd_val, name="MACD", marker_color=macd_colors, opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dif, name="DIF", line={"color": "#f0b429", "width": 1}), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dea, name="DEA", line={"color": "#f85149", "width": 1}), row=3, col=1)
    
    # 布局
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        height=500,
        margin={"l": 40, "r": 20, "t": 30, "b": 20},
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.1, x=0, font={"size": 10}),
        hovermode="x unified",
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["border"], showgrid=True, zeroline=False)
    
    # 信号卡片
    sr = tech.get("sr", {})
    mf = money_flow(code)

    signal_card = html.Div([
        html.Span("技术信号", style=HEADER_STYLE),
        html.Div([
            html.Div([
                html.Div("综合评级", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(signal.signal if signal else "N/A", style={
                    "fontSize": "20px", "fontWeight": "700",
                    "color": COLORS["green"] if signal and signal.score > 0 else COLORS["red"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("综合评分", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(f"{signal.score:+.1f}" if signal else "N/A", style={
                    "fontSize": "20px", "fontWeight": "700",
                    "color": COLORS["green"] if signal and signal.score > 0 else COLORS["red"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("置信度", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(f"{signal.confidence*100:.0f}%" if signal else "N/A", style={
                    "fontSize": "20px", "fontWeight": "700", "color": COLORS["accent"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("盈亏比", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(f"{signal.reward_risk_ratio:.2f}" if signal else "N/A", style={
                    "fontSize": "20px", "fontWeight": "700", "color": COLORS["purple"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
        ], style={"display": "flex", "marginTop": "12px"}),
        
        html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),
        
        html.Div([
            html.Div([
                html.Span("操作建议", style={"fontSize": "13px", "fontWeight": "600", "color": COLORS["accent"]}),
                html.Div(signal.advice if signal else "N/A", style={
                    "fontSize": "14px", "color": COLORS["text"], "marginTop": "4px",
                }),
            ]),
            html.Div([
                html.Span("推荐理由", style={"fontSize": "13px", "fontWeight": "600", "color": COLORS["accent"]}),
                html.Ul([html.Li(r, style={"fontSize": "12px", "marginTop": "4px"}) for r in (signal.reasons if signal else ["N/A"])]),
            ], style={"marginTop": "8px"}),
        ]),
        
        html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),
        
        # 关键价位
        html.Div([
            html.Div([
                html.Div("目标价", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
                html.Div(f"{signal.target_price:.2f}" if signal else "N/A", 
                         style={"fontSize": "16px", "fontWeight": "600", "color": COLORS["green"]}),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("止损价", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
                html.Div(f"{signal.stop_loss:.2f}" if signal else "N/A",
                         style={"fontSize": "16px", "fontWeight": "600", "color": COLORS["red"]}),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("支撑位", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
                html.Div(f"{sr.get('nearest_support', 'N/A')}",
                         style={"fontSize": "16px", "fontWeight": "600", "color": COLORS["yellow"]}),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("阻力位", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
                html.Div(f"{sr.get('nearest_resistance', 'N/A')}",
                         style={"fontSize": "16px", "fontWeight": "600", "color": COLORS["accent"]}),
            ], style={"textAlign": "center", "flex": "1"}),
        ], style={"display": "flex"}),
    ])
    
    # 资金流向卡片
    mf2 = money_flow(code)
    flow_card = html.Div([
        html.Span("资金流向分析", style=HEADER_STYLE),
        html.Div([
            html.Div([
                html.Div("主力净流入", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(f"{mf2.get('main_force', 0)/1e8:.2f}亿", style={
                    "fontSize": "20px", "fontWeight": "700",
                    "color": COLORS["green"] if mf2.get("main_force", 0) > 0 else COLORS["red"],
                }),
                html.Div(f"占比 {mf2.get('main_force_ratio', 0):+.2f}%", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("散户净流入", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(f"{mf2.get('retail', 0)/1e8:.2f}亿", style={
                    "fontSize": "20px", "fontWeight": "700",
                    "color": COLORS["text"] if mf2.get("retail", 0) >= 0 else COLORS["red"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
            html.Div([
                html.Div("信号", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Div(mf2.get("signal", ""), style={
                    "fontSize": "18px", "fontWeight": "700",
                    "color": COLORS["green"] if "流入" in mf2.get("signal", "") else COLORS["red"],
                }),
            ], style={"textAlign": "center", "flex": "1"}),
        ], style={"display": "flex", "marginTop": "12px"}),
    ])
    
    return fig, signal_card, flow_card


def _render_sectors():
    """行业轮动页面"""
    sectors = sector_quotes()
    sig_gen = SignalGenerator()
    recs = sig_gen.sector_recommendations()
    
    if sectors.empty:
        return html.Div("数据加载中...")
    
    # 行业排行图
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sectors.index.tolist(),
        y=sectors["avg_change"].values,
        marker_color=[COLORS["green"] if v >= 0 else COLORS["red"] for v in sectors["avg_change"].values],
        text=sectors["avg_change"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["card_bg"],
        plot_bgcolor=COLORS["card_bg"],
        height=400,
        margin={"l": 40, "r": 20, "t": 20, "b": 100},
        xaxis_tickangle=-45,
        yaxis_title="涨跌幅(%)",
        hovermode="x",
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    # 行业详情表
    rows = []
    for industry, rec in sorted(recs.items(), key=lambda x: x[1]["avg_score"], reverse=True):
        color = COLORS["green"] if rec["avg_score"] > 10 else COLORS["red"] if rec["avg_score"] < -10 else COLORS["text"]
        top_str = "、".join([f"{s['name']}({s['score']:+d})" for s in rec["top_stocks"]])
        rows.append(html.Tr([
            html.Td(industry, style={"padding": "6px 12px", "fontWeight": "600"}),
            html.Td(f"{rec['avg_score']:+.1f}", style={"padding": "6px 12px", "color": color, "fontWeight": "600"}),
            html.Td(rec["recommendation"], style={"padding": "6px 12px", "color": color}),
            html.Td(f"{rec['buy_count']}/{rec['sell_count']}/{rec['total']}", style={"padding": "6px 12px"}),
            html.Td(top_str, style={"padding": "6px 12px", "fontSize": "12px"}),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"}))
    
    return html.Div([
        html.Div([
            html.Span("行业轮动图", style=HEADER_STYLE),
            dcc.Graph(figure=fig, style={"height": "420px"}),
        ], style=CARD_STYLE),
        
        html.Div([
            html.Span("行业配置建议", style=HEADER_STYLE),
            html.Table(
                [html.Tr([
                    html.Th("行业", style={"padding": "8px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
                    html.Th("评分", style={"padding": "8px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
                    html.Th("建议", style={"padding": "8px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
                    html.Th("买/卖/总", style={"padding": "8px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
                    html.Th("推荐标的", style={"padding": "8px 12px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}"}),
                ])] + rows,
                style={"width": "100%", "fontSize": "13px", "color": COLORS["text"]},
            ),
        ], style=CARD_STYLE),
    ])


def _render_scan():
    """股票扫描页面"""
    sig_gen = SignalGenerator()
    top_buy = sig_gen.top_buy_signals(30)
    top_sell = sig_gen.top_sell_signals(10)
    
    def signal_row(sig, is_buy=True):
        color = COLORS["green"] if is_buy else COLORS["red"]
        return html.Tr([
            html.Td(sig.code, style={"padding": "6px 10px", "fontWeight": "600"}),
            html.Td(sig.name, style={"padding": "6px 10px"}),
            html.Td(sig.industry, style={"padding": "6px 10px", "fontSize": "12px"}),
            html.Td(sig.signal, style={"padding": "6px 10px", "color": color, "fontWeight": "600"}),
            html.Td(f"{sig.score:+.1f}", style={"padding": "6px 10px", "color": color}),
            html.Td(f"{sig.current_price:.2f}", style={"padding": "6px 10px"}),
            html.Td(f"{sig.target_price:.2f}", style={"padding": "6px 10px", "color": COLORS["green"]}),
            html.Td(f"{sig.stop_loss:.2f}", style={"padding": "6px 10px", "color": COLORS["red"]}),
            html.Td(f"{sig.upside_potential:+.1f}%", style={"padding": "6px 10px", "color": COLORS["green"]}),
            html.Td(f"{sig.reward_risk_ratio:.2f}", style={"padding": "6px 10px"}),
            html.Td(sig.reasons[0] if sig.reasons else "", style={"padding": "6px 10px", "fontSize": "11px", "maxWidth": "150px"}),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"})
    
    buy_rows = [signal_row(s, True) for s in top_buy]
    sell_rows = [signal_row(s, False) for s in top_sell]
    
    th_style = {"padding": "8px 10px", "color": COLORS["text_secondary"], "borderBottom": f"1px solid {COLORS['border']}", "fontSize": "11px", "textTransform": "uppercase"}
    
    return html.Div([
        html.Div([
            html.Span(f"买入信号 (Top {len(top_buy)})", style={**HEADER_STYLE, "color": COLORS["green"]}),
            html.Div(html.Table(
                [html.Tr([html.Th(h, style=th_style) for h in ["代码","名称","行业","信号","评分","现价","目标价","止损价","上涨空间","盈亏比","理由"]])] + buy_rows,
                style={"width": "100%", "fontSize": "12px", "color": COLORS["text"]},
            ), style={"overflow": "auto", "maxHeight": "500px"}),
        ], style=CARD_STYLE),
        
        html.Div([
            html.Span(f"卖出信号 (Top {len(top_sell)})", style={**HEADER_STYLE, "color": COLORS["red"]}),
            html.Div(html.Table(
                [html.Tr([html.Th(h, style=th_style) for h in ["代码","名称","行业","信号","评分","现价","理由"]])] + sell_rows,
                style={"width": "100%", "fontSize": "12px", "color": COLORS["text"]},
            ), style={"overflow": "auto", "maxHeight": "300px"}),
        ], style={**CARD_STYLE, "marginTop": "12px"}),
    ])




def _render_decisions():
    """投资决策页面——华尔街级别的可执行投资建议"""
    try:
        engine = InvestmentDecisionEngine()
        decisions = engine.analyze_portfolio(top_n=30)
        market = engine.market_outlook_report()
        market_state = market.get("market_state", {})
    except Exception as e:
        return html.Div(f"数据加载中... {e}", style={"color": COLORS["text_secondary"], "padding": "20px"})
    
    if not decisions:
        return html.Div("等待数据加载...", style={"color": COLORS["text_secondary"], "padding": "20px"})
    
    children = []
    
    # === 市场环境 + 组合建议 ===
    state = market_state.get("market_state", "未知")
    state_color = COLORS["green"] if state in ("牛市", "反弹") else COLORS["red"] if state in ("熊市", "回调") else COLORS["yellow"]
    
    portfolio_advice = market.get("portfolio_advice", {})
    allocation = portfolio_advice.get("allocation", {})
    
    children.append(html.Div([
        # 左侧：市场状态
        html.Div([
            html.Span("市场环境", style=HEADER_STYLE),
            html.Div(f"当前: {state}", style={
                "fontSize": "28px", "fontWeight": "700", "color": state_color,
            }),
            html.Div(f"风险等级: {market_state.get('risk_level', '中')} | 置信度: {market_state.get('confidence', 0)*100:.0f}%", 
                     style={"fontSize": "13px", "color": COLORS["text_secondary"], "marginTop": "4px"}),
            html.Div(market_state.get("action_advice", ""), 
                     style={"fontSize": "14px", "color": COLORS["text"], "marginTop": "8px", "lineHeight": "1.5"}),
        ], style={**CARD_STYLE, "flex": "1", "marginRight": "12px"}),
        
        # 右侧：组合配置建议
        html.Div([
            html.Span(f"组合策略: {portfolio_advice.get('strategy', '')}", style=HEADER_STYLE),
            html.Div([
                html.Table([
                    html.Tr([html.Td(k, style={"padding": "4px 8px", "fontSize": "13px"}), 
                             html.Td(v, style={"padding": "4px 8px", "fontSize": "13px", "fontWeight": "600", "textAlign": "right", "color": COLORS["accent"]})])
                    for k, v in allocation.items()
                ], style={"color": COLORS["text"], "width": "100%"}),
            ]),
        ], style={**CARD_STYLE, "flex": "1"}),
    ], style={"display": "flex", "marginBottom": "16px"}))
    
    # === Top 买入信号 ===
    buy_decisions = [d for d in decisions if d.composite_score > 30]
    
    children.append(html.Div([
        html.Span(f"买入信号 ({len(buy_decisions)} 只)", style={**HEADER_STYLE, "color": COLORS["green"]}),
        html.Div([
            _render_decision_card(d) for d in buy_decisions[:8]
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(340px, 1fr))", "gap": "12px"}),
    ], style={"marginBottom": "16px"}))
    
    # === 风险预警 ===
    risk_decisions = [d for d in decisions if d.composite_score < -30]
    if risk_decisions:
        children.append(html.Div([
            html.Span(f"风险预警 ({len(risk_decisions)} 只)", style={**HEADER_STYLE, "color": COLORS["red"]}),
            html.Div([
                _render_risk_card(d) for d in risk_decisions[:6]
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(300px, 1fr))", "gap": "12px"}),
        ]))
    
    return html.Div(children)


def _render_decision_card(d: InvestmentDecision):
    """渲染投资决策卡片"""
    score_color = COLORS["green"] if d.composite_score > 50 else (COLORS["yellow"] if d.composite_score > 30 else COLORS["text"])
    
    return html.Div([
        # 头部：股票名称 + 评分
        html.Div([
            html.Div([
                html.Span(d.name, style={"fontSize": "16px", "fontWeight": "700", "color": COLORS["text"]}),
                html.Span(f" {d.code}", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
            ]),
            html.Div([
                html.Span(d.decision, style={
                    "fontSize": "14px", "fontWeight": "700",
                    "color": score_color,
                    "background": f"{score_color}20",
                    "padding": "2px 10px", "borderRadius": "4px",
                }),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}),
        
        # 评分条
        html.Div([
            html.Div(style={
                "width": f"{max(0, min(100, (d.composite_score + 100) / 2))}%",
                "height": "4px",
                "background": f"linear-gradient(90deg, {COLORS['red']}, {COLORS['yellow']}, {COLORS['green']})",
                "borderRadius": "2px",
            }),
        ], style={"background": COLORS["border"], "borderRadius": "2px", "marginBottom": "8px"}),
        
        # 关键数据
        html.Div([
            html.Div([
                html.Span(f"现价: ¥{d.current_price:.2f}", style={"fontSize": "13px", "color": COLORS["text"]}),
                html.Span(f" | 目标: ¥{d.target_price_base:.2f}", style={"fontSize": "13px", "color": COLORS["accent"]}),
                html.Span(f" | 止损: ¥{d.stop_loss:.2f}", style={"fontSize": "13px", "color": COLORS["red"]}),
            ]),
            html.Div([
                html.Span(f"上涨空间: {d.expected_return:+.1f}%", style={"fontSize": "12px", "color": COLORS["green"]}),
                html.Span(f" | 盈亏比: {d.reward_risk_ratio:.1f}:1", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
                html.Span(f" | 仓位: {d.max_position_pct:.0f}%", style={"fontSize": "12px", "color": COLORS["purple"]}),
                html.Span(f" | {d.holding_period}", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
            ]),
        ], style={"marginBottom": "8px"}),
        
        # 评分明细
        html.Div([
            html.Span(f"趋势: {d.trend_score:.0f}", style={"fontSize": "11px", "color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"动量: {d.momentum_score:.0f}", style={"fontSize": "11px", "color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"估值: {d.fundamental_score:.0f}", style={"fontSize": "11px", "color": COLORS["text_secondary"], "marginRight": "8px"}),
            html.Span(f"资金: {d.capital_flow_score:.0f}", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
            html.Span(f"置信度: {d.confidence*100:.0f}%", style={"fontSize": "11px", "color": COLORS["accent"], "marginLeft": "12px"}),
        ], style={"marginBottom": "8px"}),
        
        # 摘要
        html.Div(d.summary, style={
            "fontSize": "12px", "color": COLORS["text_secondary"],
            "lineHeight": "1.5", "padding": "8px",
            "background": COLORS["bg"], "borderRadius": "4px",
        }),
    ], style={
        "background": COLORS["card_bg"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "12px",
    })


def _render_risk_card(d: InvestmentDecision):
    """渲染风险预警卡片"""
    return html.Div([
        html.Div([
            html.Span(d.name, style={"fontSize": "14px", "fontWeight": "600", "color": COLORS["red"]}),
            html.Span(f" {d.code}", style={"fontSize": "11px", "color": COLORS["text_secondary"]}),
            html.Span(f"  {d.decision}", style={
                "fontSize": "12px", "fontWeight": "600",
                "color": COLORS["red"], "float": "right",
            }),
        ]),
        html.Div([
            html.Span(f"评分: {d.composite_score:+.0f}", style={"fontSize": "12px", "color": COLORS["red"]}),
            html.Span(f" | 止损: ¥{d.stop_loss:.2f}", style={"fontSize": "12px", "color": COLORS["text_secondary"]}),
        ], style={"marginTop": "6px"}),
        html.Div(d.summary, style={
            "fontSize": "11px", "color": COLORS["text_secondary"],
            "marginTop": "4px", "lineHeight": "1.4",
        }),
    ], style={
        "background": COLORS["card_bg"],
        "border": f"1px solid {COLORS['red']}30",
        "borderRadius": "8px",
        "padding": "12px",
    })


def _render_reports():
    """投资报告页面"""
    reports = ReportGenerator.list_reports()
    
    options = [{"label": "生成新报告", "value": ""}]
    for r in reports:
        options.append({"label": f"{r['filename']} ({r['size_kb']:.1f}KB)", "value": r["path"]})
    
    return html.Div([
        html.Div([
            html.Span("已生成的报告", style=HEADER_STYLE),
            dcc.Dropdown(
                id="report-selector",
                options=options,
                value="",
                style={
                    "background": COLORS["bg"],
                    "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                },
            ),
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div([
                html.Button("生成日报", id="gen-daily", n_clicks=0, style={
                    "background": COLORS["accent"], "color": "#fff",
                    "border": "none", "borderRadius": "4px",
                    "padding": "8px 20px", "cursor": "pointer",
                    "fontWeight": "600", "marginRight": "8px",
                }),
                html.Button("生成风险预警", id="gen-risk", n_clicks=0, style={
                    "background": COLORS["yellow"], "color": "#000",
                    "border": "none", "borderRadius": "4px",
                    "padding": "8px 20px", "cursor": "pointer",
                    "fontWeight": "600",
                }),
            ]),
        ], style={**CARD_STYLE, "display": "flex", "gap": "8px", "alignItems": "center"}),
        
        html.Div(id="report-content", style={"marginTop": "12px"}),
    ])


def _render_backtest():
    """策略回测对比面板"""
    from hyperion.engine.backtest import BacktestEngine
    from hyperion.strategy.base import list_strategies
    from hyperion.data.market import CORE_STOCKS, get_stock_name
    
    engine = BacktestEngine()
    
    # Default stocks
    codes = ["600519", "000858", "300750", "601318", "000333"]
    
    # Strategy comparison for each stock
    all_rows = []
    for code in codes:
        df = engine.compare_strategies(code, days=180)
        if not df.empty:
            for _, row in df.iterrows():
                all_rows.append({
                    "stock": f"{get_stock_name(code)}({code})",
                    "strategy": row["strategy"],
                    "rating": row["rating"],
                    "annual_return": float(row["annual_return"].replace("%","")) if isinstance(row["annual_return"], str) else row["annual_return"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": float(row["max_drawdown"].replace("%","")) if isinstance(row["max_drawdown"], str) else row["max_drawdown"],
                    "win_rate": float(row["win_rate"].replace("%","")) if isinstance(row["win_rate"], str) else row["win_rate"],
                    "trades": row["total_trades"],
                    "summary": row["summary"],
                })
    
    import pandas as pd
    df_all = pd.DataFrame(all_rows)
    
    # Best performing: filter N/A and sort by sharpe
    best = df_all[df_all["rating"] != "N/A"]
    if best.empty:
        best_df = pd.DataFrame()
    else:
        best_df = best.sort_values("sharpe", ascending=False)
    
    return html.Div([
        html.H3("策略回测对比", style={"color": COLORS["text"], "marginBottom": "8px"}),
        html.P("对比 5 个核心策略在不同标的上的历史表现 (180个交易日)", 
               style={"color": COLORS["text_secondary"], "marginBottom": "16px"}),
        
        # Summary metrics
        html.Div([
            _metric_card("测试策略", f"{len(list_strategies())} 个", "多因子 / 趋势 / 均值回归 / 动量 / 异动"),
            _metric_card("测试标的", f"{len(codes)} 只", "茅台 / 五粮液 / 宁德 / 平安 / 美的"),
            _metric_card("回测区间", "180 天", "约 9 个交易月"),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),
        
        # Best strategy table
        html.Div([
            html.H4("📊 策略表现排名", style={"color": COLORS["accent"], "marginBottom": "8px"}),
            dash.dash_table.DataTable(
                data=best_df.head(15).to_dict("records"),
                columns=[
                    {"name": "标的", "id": "stock"},
                    {"name": "策略", "id": "strategy"},
                    {"name": "评级", "id": "rating"},
                    {"name": "年化收益", "id": "annual_return"},
                    {"name": "夏普", "id": "sharpe"},
                    {"name": "最大回撤", "id": "max_drawdown"},
                    {"name": "胜率", "id": "win_rate"},
                    {"name": "交易", "id": "trades"},
                ],
                style_header={
                    "backgroundColor": COLORS["card_bg"],
                    "color": COLORS["text_secondary"],
                    "border": f"1px solid {COLORS['border']}",
                    "fontWeight": "600",
                },
                style_cell={
                    "backgroundColor": COLORS["card_bg"],
                    "color": COLORS["text"],
                    "border": f"1px solid {COLORS['border']}",
                    "textAlign": "center",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{rating} = 'A+' || {rating} = 'A'"},
                        "color": "#3fb950",
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": "{rating} = 'D'"},
                        "color": "#f85149",
                    },
                    {
                        "if": {"filter_query": "{rating} = 'N/A'"},
                        "color": COLORS["text_secondary"],
                    },
                ],
                style_table={"overflowX": "auto"},
            ),
        ], style={
            "background": COLORS["card_bg"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "16px",
            "marginBottom": "20px",
        }),
        
        # Strategy interpretation
        html.Div([
            html.H4("💡 策略解读", style={"color": COLORS["accent"], "marginBottom": "8px"}),
            html.Div([
                html.P("• 趋势跟踪: 适合单边上涨/下跌行情，震荡市中容易反复打损", style={"color": COLORS["text_secondary"], "margin": "4px 0"}),
                html.P("• 均值回归: 适合震荡行情，在超卖区买入等待回归，单边市风险大", style={"color": COLORS["text_secondary"], "margin": "4px 0"}),
                html.P("• 动量突破: 适合强势市场，追涨突破信号，需要严格止损", style={"color": COLORS["text_secondary"], "margin": "4px 0"}),
                html.P("• 成交量异动: 短线策略，捕捉放量异动，持仓不超过5天", style={"color": COLORS["text_secondary"], "margin": "4px 0"}),
                html.P("• 多因子Alpha: 综合评分选股，适合中长期持仓，相对稳健", style={"color": COLORS["text_secondary"], "margin": "4px 0"}),
            ]),
        ], style={
            "background": COLORS["card_bg"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "16px",
        }),
    ])

def _render_risk():
    """风险管理页面 — VaR, 压力测试, 组合风险分析"""
    try:
        engine = InvestmentDecisionEngine()
        picks = engine.top_picks(20)
        # Build mock portfolio from top picks
        holdings = {}
        for i, d in enumerate(picks[:8]):
            holdings[d.code] = d.max_position_pct / 100.0
        total = sum(holdings.values())
        if total > 0:
            holdings = {k: v/total for k, v in holdings.items()}
        
        rm = RiskManager()
        report = rm.assess_portfolio(holdings)
    except Exception as e:
        return html.Div(f"数据加载中... {e}", style={"color": COLORS["text_secondary"], "padding": "20px"})
    
    risk_colors = {
        "low": COLORS["green"],
        "medium": COLORS["yellow"],
        "high": COLORS["red"],
        "critical": COLORS["red"],
        "unknown": COLORS["text_secondary"],
    }
    rc = risk_colors.get(report.risk_level, COLORS["text"])
    
    children = [
        # ── 风险概览仪表卡 ──
        html.Div([
            html.Div([
                html.Span("风险等级", style=HEADER_STYLE),
                html.Div(report.risk_level.upper(), style={
                    "fontSize": "32px", "fontWeight": "700", "color": rc,
                }),
                html.Div(report.advice, style={
                    "fontSize": "13px", "color": COLORS["text"], "marginTop": "8px",
                }),
            ], style={**CARD_STYLE, "flex": "1", "margin": "0 12px 0 0"}),
            
            html.Div([
                html.Span("VaR 95% (1日)", style=HEADER_STYLE),
                html.Div(f"{report.var_95_daily_pct*100:.2f}%", style={
                    "fontSize": "28px", "fontWeight": "700", "color": COLORS["text"],
                }),
                html.Div(f"99%: {report.var_99_daily*100:.2f}% | CVaR: {report.cvar_95_daily*100:.2f}%", 
                         style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginTop": "4px"}),
            ], style={**CARD_STYLE, "flex": "1", "margin": "0 12px 0 0"}),
            
            html.Div([
                html.Span("最大回撤", style=HEADER_STYLE),
                html.Div(f"{report.max_drawdown*100:.2f}%", style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": COLORS["red"] if report.max_drawdown < -0.20 else COLORS["yellow"] if report.max_drawdown < -0.10 else COLORS["green"],
                }),
                html.Div(f"当前回撤: {report.current_drawdown*100:.2f}% | 最长持续: {report.max_drawdown_duration}天",
                         style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginTop": "4px"}),
            ], style={**CARD_STYLE, "flex": "1", "margin": "0 12px 0 0"}),
            
            html.Div([
                html.Span("分散化", style=HEADER_STYLE),
                html.Div(f"有效头寸: {report.effective_n:.1f}", style={
                    "fontSize": "28px", "fontWeight": "700",
                    "color": COLORS["green"] if report.effective_n > 5 else COLORS["yellow"] if report.effective_n > 3 else COLORS["red"],
                }),
                html.Div(f"集中度 HHI: {report.concentration_risk:.3f} | Beta: {report.beta_to_market:.2f}",
                         style={"fontSize": "12px", "color": COLORS["text_secondary"], "marginTop": "4px"}),
            ], style={**CARD_STYLE, "flex": "1", "margin": "0"}),
        ], style={"display": "flex", "marginBottom": "16px"}),
        
        # ── 压力测试 ──
        html.Div([
            html.Span("压力测试情景", style=HEADER_STYLE),
            html.Div([
                html.Div([
                    html.Span(name, style={"fontSize": "13px", "color": COLORS["text"]}),
                    html.Div(style={
                        "width": "100%", "height": "6px",
                        "background": COLORS["border"],
                        "borderRadius": "3px", "marginTop": "4px",
                    }),
                    html.Div(style={
                        "width": f"{min(100, abs(loss)*100)}%" if abs(loss) < 1 else "100%",
                        "height": "6px", "marginTop": "-6px",
                        "background": COLORS["red"],
                        "borderRadius": "3px",
                    }),
                    html.Span(f"{loss*100:.0f}%", style={"fontSize": "11px", "color": COLORS["red"], "float": "right", "marginTop": "-16px"}),
                ], style={"flex": "1", "margin": "0 8px"})
                for name, loss in report.stress_tests.items()
            ], style={"display": "flex", "marginTop": "8px"}),
        ], style={**CARD_STYLE}),
        
        # ── 风险警告 ──
        html.Div([
            html.Span("风险警告与建议", style={**HEADER_STYLE, "marginBottom": "12px"}),
            *( [html.Div([
                html.Span("⚡ ", style={"fontSize": "16px"}),
                html.Span(w, style={"fontSize": "14px", "color": COLORS["yellow"]}),
            ], style={"marginBottom": "6px", "padding": "8px 12px", "background": f"{COLORS['yellow']}15", "borderRadius": "4px"})
                for w in report.warnings] if report.warnings else [
                html.Div("✓ 当前组合未检测到显著风险", style={"fontSize": "14px", "color": COLORS["green"]})
            ] ),
        ], style={**CARD_STYLE, "marginTop": "12px"}),
    ]
    
    return html.Div(children)



def run_server(host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    """启动 Dash 服务器"""
    app = create_app(debug=debug)
    if app is None:
        return
    
    print(f"\n  ✅ Hyperion Pro 仪表盘启动成功!")
    print(f"  🌐 访问地址: http://{host}:{port}")
    print(f"  ⌨  Ctrl+C 停止服务")
    print()
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
