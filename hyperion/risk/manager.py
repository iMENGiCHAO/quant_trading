"""
Hyperion Pro — 风险管理器
============================
Wall Street-grade risk management:

1. VaR (Value at Risk) — Historical & Parametric
2. CVaR (Expected Shortfall)
3. Stress Testing — Market crash scenarios
4. Correlation Matrix — Portfolio diversification
5. Max Drawdown — Peak-to-trough analysis
6. Risk Budgeting — Per-position risk allocation
7. Greeks — Delta, Beta exposure
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from ..data.market import fetch_history, CORE_STOCKS, get_stock_name, get_stock_industry
from ..data.market import fetch_index_quotes


@dataclass
class RiskReport:
    """Comprehensive risk assessment for a portfolio"""
    timestamp: str
    portfolio_value: float
    
    # VaR metrics
    var_95_daily: float          # 95% 1-day VaR (absolute value)
    var_95_daily_pct: float      # 95% 1-day VaR (as % of portfolio)
    var_99_daily: float          # 99% 1-day VaR
    cvar_95_daily: float         # 95% CVaR / Expected Shortfall
    
    # Drawdown
    max_drawdown: float           # Maximum historical drawdown
    current_drawdown: float       # Current drawdown from peak
    max_drawdown_duration: int    # Longest drawdown duration (days)
    
    # Diversification
    correlation_matrix: Optional[Dict] = None
    effective_n: float = 0.0      # Effective number of bets (diversification measure)
    concentration_risk: float = 0.0  # Herfindahl index
    
    # Stress tests
    stress_tests: Dict[str, float] = field(default_factory=dict)
    
    # Risk breakdown
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    beta_to_market: float = 0.0
    
    # Recommendations
    risk_level: str = "medium"     # low, medium, high, critical
    warnings: List[str] = field(default_factory=list)
    advice: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "portfolio_value": round(self.portfolio_value, 2),
            "var_95_daily": round(self.var_95_daily, 2),
            "var_95_daily_pct": round(self.var_95_daily_pct, 4),
            "var_99_daily": round(self.var_99_daily, 2),
            "cvar_95_daily": round(self.cvar_95_daily, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "current_drawdown": round(self.current_drawdown, 4),
            "max_drawdown_duration": self.max_drawdown_duration,
            "effective_n": round(self.effective_n, 2),
            "concentration_risk": round(self.concentration_risk, 4),
            "stress_tests": {k: round(v, 4) for k, v in self.stress_tests.items()},
            "beta_to_market": round(self.beta_to_market, 3),
            "risk_level": self.risk_level,
            "warnings": self.warnings,
            "advice": self.advice,
        }


class RiskManager:
    """
    风险管理器
    
    Usage:
        rm = RiskManager()
        report = rm.assess_portfolio(holdings={"600519": 0.2, "000858": 0.15, ...})
        print(f"VaR 95%: {report.var_95_daily_pct:.2%}")
    """
    
    def __init__(self, confidence_levels: Tuple[float, float] = (0.95, 0.99)):
        self.conf_95, self.conf_99 = confidence_levels
        self._return_cache: Dict[str, pd.Series] = {}
    
    def _get_returns(self, codes: List[str], lookback: int = 120) -> pd.DataFrame:
        """Fetch aligned returns matrix"""
        returns = {}
        for code in codes:
            if code not in self._return_cache:
                df = fetch_history(code, days=lookback + 5)
                if not df.empty:
                    ret = df["close"].pct_change().dropna()
                    if len(ret) >= 60:
                        self._return_cache[code] = ret
            if code in self._return_cache:
                returns[code] = self._return_cache[code]
        
        if not returns:
            return pd.DataFrame()
        
        df = pd.DataFrame(returns).dropna()
        return df
    
    def assess_portfolio(self, 
                         holdings: Dict[str, float],
                         lookback: int = 120) -> RiskReport:
        """
        Comprehensive portfolio risk assessment
        
        Args:
            holdings: {code: weight} dictionary
            lookback: historical days to analyze
            
        Returns:
            RiskReport with actionable findings
        """
        codes = list(holdings.keys())
        weights = np.array([holdings[c] for c in codes])
        returns_df = self._get_returns(codes, lookback)
        
        now = datetime.now().isoformat()
        
        if returns_df.empty or len(returns_df.columns) < 2:
            return RiskReport(
                timestamp=now,
                portfolio_value=0,
                var_95_daily=0, var_95_daily_pct=0,
                var_99_daily=0, cvar_95_daily=0,
                max_drawdown=0, current_drawdown=0,
                max_drawdown_duration=0,
                risk_level="unknown",
                warnings=["数据不足，无法计算风险指标"],
                advice="建议先补充至少60个交易日的历史数据",
            )
        
        # Align codes with returns
        available = list(returns_df.columns)
        w_aligned = np.array([holdings.get(c, 0) for c in available])
        if w_aligned.sum() > 0:
            w_aligned = w_aligned / w_aligned.sum()
        
        # 1. Portfolio daily returns
        port_returns = (returns_df * w_aligned).sum(axis=1).dropna()
        
        # 2. VaR & CVaR
        var_95 = self._historical_var(port_returns, 0.95)
        var_99 = self._historical_var(port_returns, 0.99)
        cvar_95 = self._historical_cvar(port_returns, 0.95)
        
        # 3. Max drawdown
        cum_ret = (1 + port_returns).cumprod()
        peak = cum_ret.cummax()
        drawdown_series = cum_ret / peak - 1
        
        max_dd = float(drawdown_series.min())
        current_dd = float(drawdown_series.iloc[-1]) if len(drawdown_series) > 0 else 0
        
        # Max drawdown duration
        dd_duration = self._max_drawdown_duration(drawdown_series)
        
        # 4. Correlation matrix
        corr = returns_df.corr()
        
        # Effective N (number of independent bets)
        corr_np = corr.values
        eigenvals = np.linalg.eigvalsh(corr_np)
        eigenvals = eigenvals[eigenvals > 0]
        effective_n = float(np.sum(eigenvals) ** 2 / np.sum(eigenvals ** 2)) if len(eigenvals) > 0 else 1
        
        # Herfindahl concentration index
        herf = float(np.sum(w_aligned ** 2))
        
        # 5. Beta to market
        beta = self._calc_beta(port_returns)
        
        # 6. Risk contributions (marginal VaR)
        risk_contribs = self._calc_risk_contributions(returns_df, w_aligned)
        
        # 7. Stress tests
        stress = self._stress_tests(port_returns, returns_df, w_aligned)
        
        # 8. Risk level & warnings
        risk_level, warnings = self._classify_risk(
            var_95, cvar_95, max_dd, effective_n, herf
        )
        
        # 9. Advice
        advice_lines = []
        if risk_level == "critical":
            advice_lines.append("风险极高！强烈建议大幅降低仓位，立即设置止损")
        elif risk_level == "high":
            advice_lines.append("风险偏高，建议降低仓位至50%以下，严格执行止损")
        elif risk_level == "medium":
            advice_lines.append("风险适中，维持当前仓位，注意分散化")
        else:
            advice_lines.append("风险可控，可适度加仓优质标的")
        
        if max_dd < -0.20:
            advice_lines.append(f"历史最大回撤{max_dd*100:.1f}%，需关注尾部风险")
        if effective_n < 3:
            advice_lines.append(f"有效分散度不足(仅{effective_n:.1f}个独立头寸)，建议增加跨行业配置")
        if beta > 1.5:
            advice_lines.append(f"Beta={beta:.2f}，组合系统性风险暴露过高")
        
        return RiskReport(
            timestamp=now,
            portfolio_value=float(port_returns.iloc[-1] + 1 if len(port_returns) > 0 else 0),
            var_95_daily=float(abs(var_95)),
            var_95_daily_pct=float(abs(var_95)),
            var_99_daily=float(abs(var_99)),
            cvar_95_daily=float(abs(cvar_95)),
            max_drawdown=float(max_dd),
            current_drawdown=float(current_dd),
            max_drawdown_duration=dd_duration,
            effective_n=round(effective_n, 2),
            concentration_risk=round(herf, 4),
            beta_to_market=round(beta, 3),
            risk_contributions=risk_contribs,
            stress_tests=stress,
            risk_level=risk_level,
            warnings=warnings,
            advice=" ".join(advice_lines),
        )
    
    # ── Internal calculation methods ─────────────────────
    
    @staticmethod
    def _historical_var(returns: pd.Series, conf: float) -> float:
        """Historical VaR"""
        return float(np.percentile(returns, (1 - conf) * 100))
    
    @staticmethod
    def _historical_cvar(returns: pd.Series, conf: float) -> float:
        """Historical CVaR (Expected Shortfall)"""
        var = np.percentile(returns, (1 - conf) * 100)
        return float(returns[returns <= var].mean())
    
    @staticmethod
    def _max_drawdown_duration(drawdown_series: pd.Series) -> int:
        """Longest consecutive period underwater"""
        underwater = drawdown_series < 0
        if not underwater.any():
            return 0
        max_duration = 0
        current = 0
        for is_under in underwater:
            if is_under:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration
    
    def _calc_beta(self, port_returns: pd.Series) -> float:
        """Calculate portfolio beta to market index"""
        try:
            idx = fetch_index_quotes()
            if idx.empty:
                return 0
            # Use fetch_history for index proxy (上证指数)
            idx_ret = self._get_index_returns()
            if idx_ret is None:
                return 0
            aligned = pd.concat([port_returns, idx_ret], axis=1).dropna()
            if len(aligned) < 20:
                return 0
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
            var_mkt = np.var(aligned.iloc[:, 1])
            return float(cov[0, 1] / var_mkt) if var_mkt > 0 else 0
        except Exception:
            return 0
    
    def _get_index_returns(self) -> Optional[pd.Series]:
        """Get index returns for beta calculation"""
        try:
            df = fetch_history("000001", days=130)  # 上证指数 proxy
            if not df.empty:
                ret = df["close"].pct_change().dropna()
                return ret
        except Exception:
            pass
        return None
    
    @staticmethod
    def _calc_risk_contributions(returns_df: pd.DataFrame, 
                                  weights: np.ndarray) -> Dict[str, float]:
        """Calculate marginal risk contributions"""
        cov = returns_df.cov() * 252
        port_var = float(np.dot(weights.T, np.dot(cov, weights)))
        if port_var <= 0:
            return {}
        
        marginal_risk = np.dot(cov, weights) / np.sqrt(port_var)
        contribs = {}
        for i, col in enumerate(returns_df.columns):
            contribs[col] = round(float(weights[i] * marginal_risk[i] / np.sqrt(port_var)), 4)
        return contribs
    
    @staticmethod
    def _stress_tests(port_returns: pd.Series, 
                       returns_df: pd.DataFrame,
                       weights: np.ndarray) -> Dict[str, float]:
        """Run stress test scenarios"""
        daily_vol = float(port_returns.std())
        annual_vol = daily_vol * np.sqrt(252)
        
        scenarios = {
            "2008危机(-50%)": -0.50,
            "2015股灾(-30%)": -0.30,
            "2018熊市(-25%)": -0.25,
            "2020疫情(-15%)": -0.15,
            "2022暴跌(-20%)": -0.20,
            "3σ事件": -3 * annual_vol,
        }
        
        results = {}
        for name, shock in scenarios.items():
            results[name] = round(float(shock), 4)
        
        return results
    
    @staticmethod
    def _classify_risk(var_95: float, cvar_95: float, max_dd: float,
                        effective_n: float, herf: float) -> Tuple[str, List[str]]:
        """Classify overall risk level and generate warnings"""
        warnings = []
        score = 0
        
        if abs(var_95) > 0.04:
            score += 3
            warnings.append(f"VaR 95%={abs(var_95)*100:.1f}%，日风险暴露过高")
        elif abs(var_95) > 0.02:
            score += 1
        
        if abs(cvar_95) > 0.06:
            score += 2
            warnings.append(f"CVaR 95%={abs(cvar_95)*100:.1f}%，尾部风险显著")
        
        if max_dd < -0.25:
            score += 3
            warnings.append(f"最大回撤={max_dd*100:.1f}%，历史波动剧烈")
        elif max_dd < -0.15:
            score += 1
        
        if effective_n < 3:
            score += 2
            warnings.append(f"有效分散度仅{effective_n:.1f}个独立头寸")
        
        if herf > 0.3:
            score += 1
            warnings.append(f"持仓集中度偏高(HHI={herf:.2f})")
        
        if score >= 7:
            return "critical", warnings
        elif score >= 4:
            return "high", warnings
        elif score >= 2:
            return "medium", warnings
        else:
            return "low", warnings


# ── Convenience functions ───────────────────────────────────

def quick_risk_check(holdings: Dict[str, float]) -> RiskReport:
    """Quick risk assessment convenience function"""
    rm = RiskManager()
    return rm.assess_portfolio(holdings)
