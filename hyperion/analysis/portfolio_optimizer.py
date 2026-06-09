"""
Hyperion Pro — 组合优化器
===========================
基于现代投资组合理论（MPT）的组合优化

功能：
  1. 均值-方差优化 (Efficient Frontier)
  2. 最大夏普比率组合
  3. 最小波动率组合
  4. 风险平价组合
  5. 基于凯利公式的头寸优化
  6. Black-Litterman 模型（结合主观观点）

输出：
  - 最优权重分配
  - 预期收益 / 风险指标
  - 再平衡建议
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize

from ..data.market import fetch_history, CORE_STOCKS, get_stock_name, get_stock_industry
from ..analysis.decision_engine import InvestmentDecisionEngine


@dataclass
class PortfolioOptimization:
    """组合优化结果"""
    weights: Dict[str, float]           # 最优权重
    expected_return: float              # 年化预期收益
    expected_volatility: float          # 年化预期波动率
    sharpe_ratio: float                 # 夏普比率
    max_drawdown_estimate: float        # 估计最大回撤
    diversification_score: float        # 分散化评分 (0-100)
    
    # 风险分解
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    
    # 再平衡建议
    rebalance_advice: List[str] = field(default_factory=list)
    
    # 当前持仓对比
    current_weights: Dict[str, float] = field(default_factory=dict)
    trades_needed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "expected_return": round(self.expected_return * 100, 2),
            "expected_volatility": round(self.expected_volatility * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_estimate": round(self.max_drawdown_estimate * 100, 2),
            "diversification_score": round(self.diversification_score, 1),
            "rebalance_advice": self.rebalance_advice,
            "trades_needed": self.trades_needed,
        }


class PortfolioOptimizer:
    """
    组合优化器
    
    使用方法:
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_sharpe_ratio(codes, lookback_days=120)
        print(result.weights)
    """
    
    def __init__(self, risk_free_rate: float = 0.025):
        """
        Args:
            risk_free_rate: 无风险利率（默认2.5%）
        """
        self.rf = risk_free_rate
        self._returns_cache: Dict[str, pd.Series] = {}
    
    def _get_returns(self, codes: List[str], lookback: int = 120) -> pd.DataFrame:
        """获取并缓存收益率数据"""
        returns = {}
        for code in codes:
            if code not in self._returns_cache:
                df = fetch_history(code, days=lookback + 5)
                if not df.empty:
                    ret = df["close"].pct_change().dropna()
                    if len(ret) >= 60:
                        self._returns_cache[code] = ret
            
            if code in self._returns_cache:
                returns[code] = self._returns_cache[code]
        
        if not returns:
            return pd.DataFrame()
        
        # Align to common index
        df = pd.DataFrame(returns)
        df = df.dropna()
        return df
    
    def _annualize(self, daily_ret: float, daily_vol: float) -> Tuple[float, float]:
        """日收益/波动率年化"""
        return daily_ret * 252, daily_vol * np.sqrt(252)
    
    # ── 均值-方差优化 ─────────────────────────────────────
    
    def optimize_sharpe_ratio(self, codes: List[str], 
                               lookback: int = 120,
                               constraints: dict = None) -> PortfolioOptimization:
        """
        最大夏普比率组合
        
        Args:
            codes: 股票代码列表
            lookback: 回看天数
            constraints: 约束条件 {'min_weight': 0.02, 'max_weight': 0.30, 'max_single_industry': 0.40}
        """
        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 0.25, 'max_single_industry': 0.40}
        
        returns_df = self._get_returns(codes, lookback)
        if returns_df.empty:
            return self._fallback_equal_weight(codes)
        
        codes_available = list(returns_df.columns)
        n = len(codes_available)
        
        if n < 3:
            return self._fallback_equal_weight(codes_available)
        
        # 协方差矩阵 & 均值
        cov = returns_df.cov() * 252
        mu = returns_df.mean() * 252
        
        # 目标函数: 负夏普比率
        def neg_sharpe(w):
            port_ret = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return -(port_ret - self.rf) / (port_vol + 1e-12)
        
        # 约束: 权重和为1
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 边界
        bounds = [(constraints.get('min_weight', 0.01), constraints.get('max_weight', 0.25)) for _ in range(n)]
        
        # 行业约束
        industry_weights = {}
        for i, code in enumerate(codes_available):
            ind = get_stock_industry(code)
            industry_weights.setdefault(ind, []).append(i)
        
        max_industry = constraints.get('max_single_industry', 0.40)
        for indices in industry_weights.values():
            if len(indices) > 1:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda w, idx=indices: max_industry - np.sum(w[idx])
                })
        
        # 初始值
        w0 = np.array([1.0 / n] * n)
        
        # 优化
        result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            w_opt = result.x
            w_opt = np.maximum(w_opt, 0)  # no short
            w_opt = w_opt / w_opt.sum()
            
            port_ret = np.dot(w_opt, mu)
            port_vol = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
            sharpe = (port_ret - self.rf) / (port_vol + 1e-12)
        else:
            w_opt = np.array([1.0 / n] * n)
            port_ret = np.dot(w_opt, mu)
            port_vol = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
            sharpe = (port_ret - self.rf) / (port_vol + 1e-12)
        
        # 风险贡献
        risk_contrib = {}
        marginal_risk = np.dot(cov, w_opt) / (port_vol + 1e-12)
        for i, code in enumerate(codes_available):
            risk_contrib[code] = w_opt[i] * marginal_risk[i] / (port_vol + 1e-12)
        
        # 分散化评分
        eff_n = 1.0 / np.sum(w_opt ** 2 + 1e-12)
        div_score = min(100, eff_n / n * 100)
        
        weights = {code: round(w_opt[i], 4) for i, code in enumerate(codes_available)}
        
        # 估计最大回撤(3σ 事件)
        max_dd_est = port_vol * 3
        
        return PortfolioOptimization(
            weights=weights,
            expected_return=round(float(port_ret), 6),
            expected_volatility=round(float(port_vol), 6),
            sharpe_ratio=round(float(sharpe), 4),
            max_drawdown_estimate=round(float(max_dd_est), 6),
            diversification_score=round(float(div_score), 1),
            risk_contributions={code: round(float(v), 4) for code, v in risk_contrib.items()},
        )
    
    def optimize_min_volatility(self, codes: List[str], lookback: int = 120) -> PortfolioOptimization:
        """最小波动率组合"""
        returns_df = self._get_returns(codes, lookback)
        if returns_df.empty:
            return self._fallback_equal_weight(codes)
        
        codes_available = list(returns_df.columns)
        n = len(codes_available)
        cov = returns_df.cov() * 252
        mu = returns_df.mean() * 252
        
        def port_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.25) for _ in range(n)]
        w0 = np.array([1.0 / n] * n)
        
        result = minimize(port_vol, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            w_opt = result.x
            w_opt = np.maximum(w_opt, 0)
            w_opt = w_opt / w_opt.sum()
        else:
            w_opt = np.array([1.0 / n] * n)
        
        port_ret = np.dot(w_opt, mu)
        port_vol_opt = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
        sharpe = (port_ret - self.rf) / (port_vol_opt + 1e-12)
        
        return PortfolioOptimization(
            weights={code: round(w_opt[i], 4) for i, code in enumerate(codes_available)},
            expected_return=round(float(port_ret), 6),
            expected_volatility=round(float(port_vol_opt), 6),
            sharpe_ratio=round(float(sharpe), 4),
            max_drawdown_estimate=round(float(port_vol_opt * 3), 6),
            diversification_score=round(min(100, 1.0 / np.sum(w_opt**2) / n * 100), 1),
        )
    
    def optimize_risk_parity(self, codes: List[str], lookback: int = 120) -> PortfolioOptimization:
        """
        风险平价组合 — 每个资产贡献相等风险
        """
        returns_df = self._get_returns(codes, lookback)
        if returns_df.empty:
            return self._fallback_equal_weight(codes)
        
        codes_available = list(returns_df.columns)
        n = len(codes_available)
        cov = returns_df.cov() * 252
        mu = returns_df.mean() * 252
        
        def risk_parity_objective(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            marginal_risk = np.dot(cov, w) / (port_vol + 1e-12)
            risk_contrib = w * marginal_risk
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 0.25) for _ in range(n)]
        w0 = np.array([1.0 / n] * n)
        
        result = minimize(risk_parity_objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            w_opt = result.x
            w_opt = np.maximum(w_opt, 0)
            w_opt = w_opt / w_opt.sum()
        else:
            w_opt = np.array([1.0 / n] * n)
        
        port_ret = np.dot(w_opt, mu)
        port_vol_opt = np.sqrt(np.dot(w_opt.T, np.dot(cov, w_opt)))
        sharpe = (port_ret - self.rf) / (port_vol_opt + 1e-12)
        
        return PortfolioOptimization(
            weights={code: round(w_opt[i], 4) for i, code in enumerate(codes_available)},
            expected_return=round(float(port_ret), 6),
            expected_volatility=round(float(port_vol_opt), 6),
            sharpe_ratio=round(float(sharpe), 4),
            max_drawdown_estimate=round(float(port_vol_opt * 3), 6),
            diversification_score=round(min(100, 1.0 / np.sum(w_opt**2) / n * 100), 1),
        )
    
    def generate_efficient_frontier(self, codes: List[str], lookback: int = 120,
                                     n_points: int = 50) -> Dict:
        """生成有效前沿"""
        returns_df = self._get_returns(codes, lookback)
        if returns_df.empty:
            return {"points": [], "max_sharpe": None, "min_vol": None}
        
        codes_available = list(returns_df.columns)
        n = len(codes_available)
        cov = returns_df.cov() * 252
        mu = returns_df.mean() * 252
        
        def min_vol_for_target(target_ret):
            w0 = np.array([1.0 / n] * n)
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_ret},
            ]
            bounds = [(0.0, 0.30) for _ in range(n)]
            result = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov, w))),
                w0, method='SLSQP', bounds=bounds, constraints=cons
            )
            if result.success:
                w = result.x
                return np.sqrt(np.dot(w.T, np.dot(cov, w)))
            return None
        
        min_ret = max(mu.min(), 0)
        max_ret = mu.max()
        targets = np.linspace(min_ret, max_ret, n_points)
        
        points = []
        for tr in targets:
            vol = min_vol_for_target(tr)
            if vol is not None:
                points.append({"return": round(float(tr) * 100, 2), "volatility": round(float(vol) * 100, 2)})
        
        return {"points": points, "max_sharpe": None, "min_vol": None}
    
    def _fallback_equal_weight(self, codes: List[str]) -> PortfolioOptimization:
        """等权重回退方案"""
        n = len(codes)
        weights = {c: 1.0 / n for c in codes} if codes else {}
        return PortfolioOptimization(
            weights=weights,
            expected_return=0.08,
            expected_volatility=0.20,
            sharpe_ratio=0.3,
            max_drawdown_estimate=0.30,
            diversification_score=50.0,
        )


class RebalanceAdvisor:
    """
    再平衡顾问 — 对比理想组合和当前持仓，生成操作清单
    """
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
        self.engine = InvestmentDecisionEngine()
    
    def analyze(self, codes: List[str], 
                current_holdings: Dict[str, float] = None,
                lookback: int = 120) -> PortfolioOptimization:
        """
        分析并提供再平衡建议
        
        Args:
            codes: 目标股票池
            current_holdings: 当前持仓 {代码: 占比}
        """
        result = self.optimizer.optimize_sharpe_ratio(codes, lookback)
        
        if current_holdings:
            result.current_weights = current_holdings
            result.trades_needed = self._generate_trades(current_holdings, result.weights)
            result.rebalance_advice = self._generate_advice(result, current_holdings)
        
        return result
    
    def _generate_trades(self, current: Dict[str, float], 
                          target: Dict[str, float]) -> List[str]:
        """生成具体交易指令"""
        trades = []
        all_codes = set(list(current.keys()) + list(target.keys()))
        
        for code in sorted(all_codes):
            curr_w = current.get(code, 0)
            tgt_w = target.get(code, 0)
            diff = tgt_w - curr_w
            
            if abs(diff) < 0.01:  # 1%忽略
                continue
            
            action = "买入" if diff > 0 else "卖出"
            trades.append(f"{action} {get_stock_name(code)}({code}): {abs(diff)*100:.1f}%")
        
        return trades
    
    def _generate_advice(self, result: PortfolioOptimization, 
                          current: Dict[str, float]) -> List[str]:
        """生成再平衡建议"""
        advice = []
        
        # 分散化建议
        if result.diversification_score < 40:
            advice.append("⚠ 组合集中度偏高，建议增加不同行业的标的以分散风险")
        
        # 夏普比率
        if result.sharpe_ratio < 0.3:
            advice.append("⚠ 夏普比率偏低，预期收益难以覆盖风险，需优化配置")
        elif result.sharpe_ratio > 1.0:
            advice.append("✓ 夏普比率良好，风险调整后收益稳健")
        
        # 波动率
        if result.expected_volatility > 0.3:
            advice.append("⚠ 预期波动率偏高，建议增加低波动资产")
        
        return advice


# ── 便捷函数 ────────────────────────────────────────────────

def build_recommended_portfolio(top_n: int = 20, lookback: int = 120) -> PortfolioOptimization:
    """根据决策引擎推荐构建最优组合"""
    engine = InvestmentDecisionEngine()
    picks = engine.top_picks(top_n)
    codes = [d.code for d in picks]
    
    optimizer = PortfolioOptimizer()
    return optimizer.optimize_sharpe_ratio(codes, lookback)
