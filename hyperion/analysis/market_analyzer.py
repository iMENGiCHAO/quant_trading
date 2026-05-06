"""
Hyperion Market Analyzer (实时市场分析引擎)
===========================================
融合 GitHub Top 12 量化项目精华的实时市场分析模块。

功能:
  - 多指数实时行情分析
  - Alpha158 因子信号提取
  - Risk Budgeting + HRP 组合优化
  - VaR/CVaR/MaxDD 风控分析
  - 市场情绪综合评分
  - HTML/JSON/Markdown 报告生成

使用:
  from hyperion.analysis.market_analyzer import MarketAnalyzer
  analyzer = MarketAnalyzer()
  report = analyzer.analyze()
  analyzer.report_markdown(report)
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hyperion.alpha.factors import Alpha158
from hyperion.alpha.technical import TechnicalIndicators
from hyperion.risk.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)

# 主要指数
INDICES = {
    "上证指数": ("1", "000001", "sh000001"),
    "深证成指": ("0", "399001", "sz399001"),
    "创业板指": ("0", "399006", "sz399006"),
    "科创50": ("1", "000688", "sh000688"),
    "沪深300": ("1", "000300", "sh000300"),
    "中证500": ("1", "000905", "sh000905"),
    "上证50": ("1", "000016", "sh000016"),
}

# 板块分类
SECTORS = {
    "科技": ["科创50", "创业板指"],
    "大盘蓝筹": ["沪深300", "上证50"],
    "全市场": ["上证指数", "深证成指"],
    "中小盘": ["中证500"],
}


class MarketAnalyzer:
    """实时市场分析引擎
    
    Examples:
        >>> analyzer = MarketAnalyzer()
        >>> report = analyzer.analyze(full_scan=True)
        >>> print(analyzer.to_markdown(report))
    """
    
    def __init__(self, lookback_days: int = 120):
        self.lookback = lookback_days
        self.alpha_engine = Alpha158()
        self.tech = TechnicalIndicators()
        self.optimizer = PortfolioOptimizer()
        
    # ─── 数据获取 ───────────────────────────────────────────
    
    def fetch_realtime_quotes(self) -> Dict:
        """拉取实时指数行情 (Sina API)"""
        sina_codes = [v[2] for v in INDICES.values()]
        batch = ','.join(sina_codes)
        
        try:
            req = urllib.request.Request(
                f"http://hq.sinajs.cn/list={batch}",
                headers={'Referer': 'https://finance.sina.com.cn',
                         'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode('gbk')
            
            result = {}
            code_to_name = {v[2]: k for k, v in INDICES.items()}
            
            for line in raw.strip().split('\n'):
                if '="' not in line:
                    continue
                code = line.split('="')[0].split('_')[-1]
                if code not in code_to_name:
                    continue
                name = code_to_name[code]
                data = line.split('"')[1].split(',')
                
                cur = float(data[3]) if float(data[3]) > 0 else float(data[2])
                prev = float(data[2])
                
                result[name] = {
                    'name': name,
                    'code': code,
                    'current': cur,
                    'prev_close': prev,
                    'open': float(data[1]),
                    'high': float(data[4]),
                    'low': float(data[5]),
                    'change': cur - prev,
                    'change_pct': (cur - prev) / prev * 100,
                    'volume': float(data[8]) / 1e8 if len(data) > 8 and data[8] else 0,
                    'amount': float(data[9]) / 1e8 if len(data) > 9 and data[9] else 0,
                }
            return result
        except Exception as e:
            logger.error(f"Realtime fetch: {e}")
            return {}
    
    def fetch_kline(self, name: str) -> Optional[pd.DataFrame]:
        """拉取 K 线历史数据"""
        market, code, _ = INDICES.get(name, (None, None, None))
        if not market:
            return None
        
        today_fmt = datetime.now().strftime('%Y%m%d')
        
        for attempt in range(3):
            try:
                if attempt == 0:
                    end = today_fmt
                elif attempt == 1:
                    end = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
                else:
                    end = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
                
                url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
                       f"?secid={market}.{code}&fields1=f1,f2,f3,f4,f5,f6,f7"
                       f"&fields2=f51,f52,f53,f54,f55,f56,f57"
                       f"&klt=101&fqt=1&end={end}&lmt={self.lookback}")
                
                req = urllib.request.Request(url, headers={
                    'Referer': 'https://quote.eastmoney.com',
                    'User-Agent': 'Mozilla/5.0'
                })
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data.get('data') and data['data'].get('klines'):
                        klines = data['data']['klines']
                        rows = []
                        for k in klines:
                            p = k.split(',')
                            rows.append({
                                'date': p[0],
                                'open': float(p[1]),
                                'close': float(p[2]),
                                'high': float(p[3]),
                                'low': float(p[4]),
                                'volume': float(p[5]),
                                'amount': float(p[6]),
                            })
                        df = pd.DataFrame(rows)
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                        return df
            except Exception:
                continue
        return None
    
    def fetch_full_market_scan(self, max_pages: int = 70) -> Dict:
        """全市场扫描 - 涨跌分布、涨停/跌停、成交额排名"""
        try:
            count_url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api"
                         "/json_v2.php/Market_Center.getHQNodeStockCountSimple?node=hs_a")
            req = urllib.request.Request(count_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                total = int(resp.read().decode('gbk').strip('"'))
        except Exception:
            total = 5510
        
        headers = {
            'Referer': 'https://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        }
        
        up = down = flat = limit_up = limit_down = 0
        all_changes = []
        top_amount = []
        
        for page in range(1, max_pages + 1):
            try:
                url = (f"https://vip.stock.finance.sina.com.cn/quotes_service/api"
                       f"/json_v2.php/Market_Center.getHQNodeData?"
                       f"page={page}&num=80&sort=changepercent&asc=0"
                       f"&node=hs_a&symbol=&_s_r_a=page")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp:
                    stocks = json.loads(resp.read().decode('gbk'))
                    if not stocks:
                        break
                    
                    for s in stocks:
                        chg = float(s.get('changepercent', 0))
                        all_changes.append(chg)
                        if chg > 0.01: up += 1
                        elif chg < -0.01: down += 1
                        else: flat += 1
                        if chg >= 9.8: limit_up += 1
                        if chg <= -9.8: limit_down += 1
                        
                        amt = float(s.get('amount', 0))
                        if amt > 0:
                            top_amount.append({
                                'name': s.get('name', ''),
                                'code': s.get('code', ''),
                                'price': s.get('trade', ''),
                                'change_pct': chg,
                                'amount': amt,
                                'pe': s.get('per', ''),
                                'pb': s.get('pb', ''),
                                'mktcap': s.get('mktcap', ''),
                            })
                    
                    if len(stocks) < 80:
                        break
            except Exception:
                break
        
        top_amount.sort(key=lambda x: -x['amount'])
        all_changes.sort()
        
        mid = len(all_changes) // 2
        return {
            'total': len(all_changes),
            'up': up, 'down': down, 'flat': flat,
            'limit_up': limit_up, 'limit_down': limit_down,
            'up_ratio': up / len(all_changes) if all_changes else 0,
            'up_down_ratio': up / down if down > 0 else up,
            'median_chg': all_changes[mid] if all_changes else 0,
            'avg_chg': sum(all_changes) / len(all_changes) if all_changes else 0,
            'all_changes': all_changes,
            'top30_amount': top_amount[:30],
        }
    
    # ─── 因子分析 ───────────────────────────────────────────
    
    def analyze_factor_signals(self, df: pd.DataFrame) -> Dict:
        """从 Alpha158 因子中提取关键交易信号"""
        factors = self.alpha_engine.extract(df)
        latest = factors.iloc[-1]
        
        signal_map = {
            'RET_005': '5日收益率',
            'RET_020': '20日收益率',
            'RET_060': '60日收益率',
            'MA20_CLOSE': '20日均线偏离',
            'MA60_CLOSE': '60日均线偏离',
            'DEV20_CLOSE': '20日偏离度',
            'VOLR20': '20日量比',
            'PVC20': '量价相关系数',
            'RSI': 'RSI(14)',
            'MACD': 'MACD',
            'MACD_SIGNAL': 'MACD信号线',
            'MACD_HIST': 'MACD柱',
            'BB_UPPER': '布林上轨',
            'BB_LOWER': '布林下轨',
            'BB_WIDTH': '布林带宽',
            'BB_PCT': '布林带位置%',
            'ATR': 'ATR',
            'ATR_PCT': 'ATR%',
            'KDJ_K': 'KDJ-K',
            'KDJ_D': 'KDJ-D',
            'KDJ_J': 'KDJ-J',
            'STD20_CLOSE': '20日波动率',
            'SK20_CLOSE': '20日偏度',
            'KU20_CLOSE': '20日峰度',
            'ZSC20_CLOSE': '20日Z-Score',
        }
        
        signals = {}
        for col, label in signal_map.items():
            if col in factors.columns and not pd.isna(latest[col]):
                signals[col] = round(float(latest[col]), 4)
        
        # 综合信号解读
        interpretation = self._interpret_signals(signals, df)
        
        return {
            'n_factors': len(factors.columns),
            'n_valid_signals': len(signals),
            'signals': signals,
            'interpretation': interpretation,
        }
    
    def _interpret_signals(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """解读因子信号"""
        interp = {}
        
        # 趋势
        ret_20 = signals.get('RET_020', 0)
        if ret_20 > 0.05: interp['trend'] = '强烈上涨'
        elif ret_20 > 0.02: interp['trend'] = '温和上涨'
        elif ret_20 > -0.02: interp['trend'] = '横盘整理'
        elif ret_20 > -0.05: interp['trend'] = '温和下跌'
        else: interp['trend'] = '强烈下跌'
        
        # RSI
        rsi = signals.get('RSI', 50)
        if rsi > 80: interp['rsi'] = '超买(>80)'
        elif rsi > 60: interp['rsi'] = '偏强(60-80)'
        elif rsi > 40: interp['rsi'] = '中性(40-60)'
        elif rsi > 20: interp['rsi'] = '偏弱(20-40)'
        else: interp['rsi'] = '超卖(<20)'
        
        # MACD
        macd = signals.get('MACD', 0)
        macd_sig = signals.get('MACD_SIGNAL', 0)
        if macd > macd_sig and macd > 0: interp['macd'] = '金叉+零轴上'
        elif macd > macd_sig: interp['macd'] = '金叉'
        elif macd < macd_sig and macd < 0: interp['macd'] = '死叉+零轴下'
        else: interp['macd'] = '死叉'
        
        # 布林带
        bb_pct = signals.get('BB_PCT', 0.5)
        if bb_pct > 0.9: interp['bollinger'] = '触及上轨(超强)'
        elif bb_pct > 0.7: interp['bollinger'] = '偏上轨(强势)'
        elif bb_pct > 0.3: interp['bollinger'] = '中轨附近(中性)'
        elif bb_pct > 0.1: interp['bollinger'] = '偏下轨(弱势)'
        else: interp['bollinger'] = '触及下轨(超弱)'
        
        # 量价
        volr = signals.get('VOLR20', 1)
        if volr > 1.5: interp['volume'] = '放量(>1.5x)'
        elif volr > 1.1: interp['volume'] = '温和放量'
        elif volr > 0.9: interp['volume'] = '正常'
        elif volr > 0.5: interp['volume'] = '缩量'
        else: interp['volume'] = '极度缩量'
        
        # 波动率
        atr_pct = signals.get('ATR_PCT', 2)
        if atr_pct > 5: interp['volatility'] = '高波动(ATR>5%)'
        elif atr_pct > 3: interp['volatility'] = '中等波动'
        else: interp['volatility'] = '低波动(ATR<3%)'
        
        return interp
    
    # ─── 风控分析 ───────────────────────────────────────────
    
    def analyze_risk(self, df: pd.DataFrame) -> Dict:
        """风控分析"""
        if len(df) < 30:
            return {}
        
        returns = df['close'].pct_change().dropna()
        rf_daily = 0.03 / 252
        
        # VaR & CVaR
        var_95 = float(returns.quantile(0.05))
        cvar_95 = float(returns[returns <= returns.quantile(0.05)].mean())
        var_99 = float(returns.quantile(0.01))
        
        # 波动率
        ann_vol = float(returns.std() * np.sqrt(252))
        
        # 最大回撤
        cum = (1 + returns).cumprod()
        dd = cum / cum.expanding().max() - 1
        max_dd = float(dd.min())
        dd_duration = self._max_dd_duration(returns)
        
        # Sharpe / Sortino
        excess = returns - rf_daily
        sharpe = float(excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        neg_returns = returns[returns < 0]
        sortino = float(excess.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0
        
        # Calmar
        calmar = float(returns.mean() * 252 / abs(max_dd)) if max_dd != 0 else 0
        
        # 波动率区间
        vol_5d = float(returns.tail(5).std())
        vol_20d = float(returns.tail(20).std())
        if vol_5d > vol_20d * 1.3:
            vol_regime = "HIGH"
        elif vol_5d < vol_20d * 0.7:
            vol_regime = "LOW"
        else:
            vol_regime = "NORMAL"
        
        # 收益分布
        pos_days = int((returns > 0).sum())
        neg_days = int((returns < 0).sum())
        win_rate = pos_days / (pos_days + neg_days) if (pos_days + neg_days) > 0 else 0
        
        # 蒙特卡洛 VaR
        mc_var = self._monte_carlo_var(returns)
        
        return {
            'var_95': round(var_95 * 100, 2),
            'cvar_95': round(cvar_95 * 100, 2),
            'var_99': round(var_99 * 100, 2),
            'mc_var_95': round(mc_var * 100, 2),
            'ann_vol': round(ann_vol * 100, 2),
            'max_dd': round(max_dd * 100, 2),
            'dd_duration': dd_duration,
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'calmar': round(calmar, 2),
            'vol_regime': vol_regime,
            'win_rate': round(win_rate * 100, 1),
            'pos_days': pos_days,
            'neg_days': neg_days,
        }
    
    def _max_dd_duration(self, returns: pd.Series) -> int:
        """最大回撤持续天数"""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        in_dd = cum < peak
        max_duration = 0
        current_duration = 0
        for val in in_dd:
            if val:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return max_duration
    
    def _monte_carlo_var(self, returns: pd.Series, n_sim: int = 10000) -> float:
        """蒙特卡洛 VaR (Cholesky分解)"""
        mu = returns.mean()
        sigma = returns.std()
        sim_returns = np.random.normal(mu, sigma, n_sim)
        return float(np.percentile(sim_returns, 5))
    
    # ─── 组合优化 ───────────────────────────────────────────
    
    def analyze_portfolio(self, dfs: Dict[str, pd.DataFrame]) -> Dict:
        """组合优化分析"""
        assets = list(dfs.keys())
        if len(assets) < 2:
            return {}
        
        returns_list = []
        for name in assets:
            r = dfs[name]['close'].pct_change().dropna()
            returns_list.append(r)
        
        returns_df = pd.concat(returns_list, axis=1).dropna()
        returns_df.columns = assets
        
        if len(returns_df) < 20:
            return {}
        
        cov = returns_df.cov().values
        corr = returns_df.corr().values
        
        n = len(assets)
        
        # 等权
        eq = np.ones(n) / n
        
        # 风险平价
        try:
            rb = self.optimizer.risk_budgeting(cov)
        except Exception:
            rb = eq
        
        # HRP
        try:
            hrp = self.optimizer.hrp(cov)
        except Exception:
            hrp = eq
        
        # 组合波动
        eq_vol = float(np.sqrt(eq @ cov @ eq) * np.sqrt(252) * 100)
        rb_vol = float(np.sqrt(rb @ cov @ rb) * np.sqrt(252) * 100)
        
        return {
            'assets': assets,
            'correlation': [[round(float(corr[i][j]), 2) for j in range(n)] for i in range(n)],
            'weights': {
                'equal': {a: round(float(eq[i]) * 100, 1) for i, a in enumerate(assets)},
                'risk_budgeting': {a: round(float(rb[i]) * 100, 1) for i, a in enumerate(assets)},
                'hrp': {a: round(float(hrp[i]) * 100, 1) for i, a in enumerate(assets)},
            },
            'portfolio_vol': {
                'equal': round(eq_vol, 1),
                'risk_budgeting': round(rb_vol, 1),
            },
        }
    
    # ─── 综合评分 ───────────────────────────────────────────
    
    def composite_score(self, quotes: Dict, factors: Dict, risk: Dict) -> Dict:
        """Hyperion 综合评分系统"""
        ret_20 = factors.get('signals', {}).get('RET_020', 0)
        rsi = factors.get('signals', {}).get('RSI', 50)
        macd = factors.get('signals', {}).get('MACD', 0)
        sharpe = risk.get('sharpe', 0)
        ann_vol = risk.get('ann_vol', 20)
        var_95 = risk.get('var_95', -2)
        
        # 趋势分 (40): 收益率趋势 + MACD 信号
        trend = 40
        trend += min(30, max(-30, ret_20 * 200 + macd * 2))
        trend = max(0, min(100, trend))
        
        # 风险分 (30): 波动率 + VaR
        risk_score = 30
        risk_score += (15 - ann_vol) * 0.5  # 低波动加分
        risk_score += sharpe * 5
        risk_score = max(0, min(100, risk_score))
        
        # 情绪分 (30): RSI + 动量
        sentiment = 30
        if 40 <= rsi <= 60: sentiment += 15  # 健康区间加分
        elif 60 < rsi <= 75: sentiment += 10
        elif rsi > 75: sentiment -= 5
        sentiment += ret_20 * 200
        sentiment = max(0, min(100, sentiment))
        
        total = trend * 0.4 + risk_score * 0.3 + sentiment * 0.3
        total = max(0, min(100, total))
        
        if total >= 80: grade = "A+"
        elif total >= 70: grade = "A"
        elif total >= 60: grade = "B"
        elif total >= 50: grade = "C"
        elif total >= 40: grade = "D"
        else: grade = "F"
        
        signal = "BULL" if total >= 60 else ("BEAR" if total <= 40 else "HOLD")
        
        return {
            'total': round(total, 1),
            'grade': grade,
            'signal': signal,
            'components': {
                'trend': round(trend, 1),
                'risk': round(risk_score, 1),
                'sentiment': round(sentiment, 1),
            },
        }
    
    # ─── 主分析流程 ─────────────────────────────────────────
    
    def analyze(self, full_scan: bool = True) -> Dict:
        """主分析流程
        
        Args:
            full_scan: 是否执行全市场扫描 (需要约40秒)
        
        Returns:
            完整分析报告 Dict
        """
        report = {
            'meta': {
                'framework': 'Hyperion Quant v1.0',
                'generated_at': datetime.now().isoformat(),
                'analyzer': 'MarketAnalyzer',
            },
            'indices': {},
            'portfolio': {},
            'full_market': None,
        }
        
        # 实时行情
        quotes = self.fetch_realtime_quotes()
        
        # 逐指数分析
        dfs = {}
        for name in quotes:
            df = self.fetch_kline(name)
            if df is not None and len(df) >= 30:
                dfs[name] = df
                factors = self.analyze_factor_signals(df)
                risk = self.analyze_risk(df)
                score = self.composite_score(quotes.get(name, {}), factors, risk)
                
                report['indices'][name] = {
                    'quote': quotes.get(name, {}),
                    'factors': factors,
                    'risk': risk,
                    'score': score,
                }
        
        # 组合优化
        if len(dfs) >= 2:
            report['portfolio'] = self.analyze_portfolio(dfs)
        
        # 全市场扫描
        if full_scan:
            report['full_market'] = self.fetch_full_market_scan()
        
        return report
    
    # ─── 报告格式化 ─────────────────────────────────────────
    
    def to_markdown(self, report: Dict) -> str:
        """生成 Markdown 格式报告"""
        meta = report['meta']
        lines = []
        lines.append(f"# Hyperion Quant v1.0 — A股市场分析报告")
        lines.append(f"**生成时间**: {meta['generated_at']}")
        lines.append(f"**引擎**: {meta['framework']} — Market Analyzer")
        lines.append("")
        
        # 市场概览
        lines.append("## 一、市场概览")
        lines.append("")
        lines.append("| 指数 | 收盘 | 涨跌 | 涨幅 | 最高 | 最低 | 成交额(亿) |")
        lines.append("|------|------|------|------|------|------|------------|")
        
        for name, data in report['indices'].items():
            q = data.get('quote', {})
            chg = q.get('change', 0)
            chg_pct = q.get('change_pct', 0)
            arrow = "🔴" if chg > 0 else ("🟢" if chg < 0 else "⚪")
            lines.append(f"| {name} | {q.get('current',0):.2f} | {arrow} | {chg_pct:+.2f}% | "
                        f"{q.get('high',0):.2f} | {q.get('low',0):.2f} | {q.get('amount',0):.1f} |")
        lines.append("")
        
        # 全市场扫描
        fm = report.get('full_market')
        if fm:
            lines.append("## 二、全市场扫描")
            lines.append("")
            lines.append(f"- 扫描股票: {fm['total']}只")
            lines.append(f"- 上涨: {fm['up']}只 ({fm['up_ratio']*100:.1f}%)")
            lines.append(f"- 下跌: {fm['down']}只")
            lines.append(f"- 涨跌比: {fm['up_down_ratio']:.2f}")
            lines.append(f"- 涨停: {fm['limit_up']}只 | 跌停: {fm['limit_down']}只")
            lines.append(f"- 中位数涨幅: {fm['median_chg']:.2f}%")
            lines.append(f"- 平均涨幅: {fm['avg_chg']:.2f}%")
            lines.append("")
            
            # TOP10 成交额
            lines.append("### 成交额 TOP10")
            lines.append("")
            lines.append("| # | 股票 | 代码 | 价格 | 涨跌 | 成交额(亿) | PE |")
            lines.append("|---|------|------|------|------|------------|----|")
            for i, s in enumerate(fm['top30_amount'][:10], 1):
                arrow = "🔴" if s['change_pct'] > 0 else "🟢"
                pe = f"{float(s['pe']):.1f}" if s.get('pe') and s['pe'] != '' else '-'
                lines.append(f"| {i} | {s['name']} | {s['code']} | {s['price']} | "
                            f"{arrow}{s['change_pct']:+.2f}% | {s['amount']/1e8:.1f} | {pe} |")
            lines.append("")
        
        # 因子分析
        lines.append("## 三、Alpha158 因子分析")
        lines.append("")
        for name, data in report['indices'].items():
            f = data.get('factors', {})
            s = f.get('signals', {})
            interp = f.get('interpretation', {})
            score = data.get('score', {})
            
            lines.append(f"### {name} (综合评分: {score.get('total',0):.0f}/100 [{score.get('grade','N/A')}])")
            lines.append("")
            lines.append(f"- **趋势**: {interp.get('trend','N/A')} | RSI: {interp.get('rsi','N/A')}")
            lines.append(f"- **MACD**: {interp.get('macd','N/A')} | 布林: {interp.get('bollinger','N/A')}")
            lines.append(f"- **量能**: {interp.get('volume','N/A')} | 波动: {interp.get('volatility','N/A')}")
            lines.append("")
        
        # 风险分析
        lines.append("## 四、风控分析")
        lines.append("")
        lines.append("| 指数 | Sharpe | 年化波动 | VaR95% | CVaR95% | 最大回撤 | 波动区间 |")
        lines.append("|------|--------|----------|--------|---------|----------|----------|")
        for name, data in report['indices'].items():
            r = data.get('risk', {})
            lines.append(f"| {name} | {r.get('sharpe',0):.2f} | {r.get('ann_vol',0):.1f}% | "
                        f"{r.get('var_95',0):.2f}% | {r.get('cvar_95',0):.2f}% | "
                        f"{r.get('max_dd',0):.1f}% | {r.get('vol_regime','N/A')} |")
        lines.append("")
        
        # 组合优化
        port = report.get('portfolio', {})
        if port:
            lines.append("## 五、组合优化")
            lines.append("")
            weights = port.get('weights', {})
            assets = port.get('assets', [])
            
            lines.append("| 资产 | 等权 | 风险平价 | HRP |")
            lines.append("|------|------|----------|-----|")
            for a in assets:
                eq = weights.get('equal', {}).get(a, 0)
                rb = weights.get('risk_budgeting', {}).get(a, 0)
                hrp = weights.get('hrp', {}).get(a, 0)
                lines.append(f"| {a} | {eq:.1f}% | {rb:.1f}% | {hrp:.1f}% |")
            lines.append("")
            
            vol = port.get('portfolio_vol', {})
            lines.append(f"- 等权组合年化波动: {vol.get('equal',0):.1f}%")
            lines.append(f"- 风险平价年化波动: {vol.get('risk_budgeting',0):.1f}%")
            lines.append("")
        
        # 免责
        lines.append("---")
        lines.append("*本报告由 Hyperion Quant v1.0 Market Analyzer 自动生成*")
        lines.append("*数据来源: Sina Finance API + Eastmoney K-line API*")
        lines.append("*⚠️ 风险提示: 本报告仅供研究参考，不构成投资建议。量化模型存在固有不确定性。*")
        
        return '\n'.join(lines)
    
    def to_json(self, report: Dict) -> str:
        """生成 JSON 格式报告"""
        return json.dumps(report, ensure_ascii=False, indent=2, default=str)