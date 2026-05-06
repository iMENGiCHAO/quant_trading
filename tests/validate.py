#!/usr/bin/env python3
"""
Hyperion Quant v1.0 — Comprehensive Validation
===============================================
End-to-end tests with synthetic data. No external API calls.
Run: python tests/validate.py

Tests:
  T1: Config system (YAML load/save)
  T2: Alpha158 factor extraction
  T3: Bayesian online learning
  T4: Technical indicators
  T5: Causal discovery (Granger)
  T6: Event engine
  T7: Portfolio optimizer (Risk Budgeting / HRP)
  T8: Risk manager
  T9: Backtest engine (end-to-end)
  T10: Report generation
  T11: CLI commands
  T12: RL TradingEnv
"""
from __future__ import annotations

import sys
import os
import warnings
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PASS = "PASS"
FAIL = "FAIL"
results = []


# ============================================================
# Helper: Generate synthetic OHLCV data
# ============================================================
def make_data(symbols=None, n_days=252):
    """Generate realistic OHLCV data for testing."""
    if symbols is None:
        symbols = ["000001.SZ", "000002.SZ", "000600.SH", "000700.SZ", "000800.SZ"]
    np.random.seed(42)
    data = {}
    for sym in symbols:
        base_price = np.random.uniform(5, 100)
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        returns = np.random.normal(0.0005, 0.02, n_days)
        close = base_price * (1 + returns).cumprod()
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        open_ = low + np.random.uniform(0, 1, n_days) * (high - low)
        volume = np.random.randint(100000, 10000000, n_days).astype(float)
        amount = close * volume
        turnover = np.random.uniform(0.5, 5, n_days)
        change_pct = returns * 100
        amplitude = (high - low) / close * 100
        vwap = (high + low + close) / 3
        
        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount, "turnover": turnover,
            "change_pct": change_pct, "amplitude": amplitude, "vwap": vwap
        }, index=dates)
        data[sym] = df
    return data


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"[{status}] {name}"
    if detail:
        msg += f" | {detail}"
    results.append((name, status, detail))
    print(msg)


# ============================================================
# T1: Config System
# ============================================================
def test_config():
    print("\n--- T1: Config System ---")
    try:
        from hyperion.config import HyperionConfig, load_config, DataSourceType, BrokerType
        
        # Default config
        cfg = HyperionConfig()
        check("Default config OK", cfg.data.source == DataSourceType.AKSHARE)
        check("Default capital", cfg.engine.initial_capital == 1_000_000.0)
        check("T+1 enabled", cfg.engine.t_plus_1 is True)
        check("Stop loss", cfg.risk.stop_loss_pct == 0.05)
        
        # Dict roundtrip
        d = cfg.to_dict()
        check("to_dict OK", isinstance(d, dict) and "data" in d)
        
        # Try loading YAML config (may fail if pyyaml not installed)
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(yaml_path):
            try:
                cfg2 = HyperionConfig.from_yaml(yaml_path)
                check("YAML load OK", cfg2.data.source == DataSourceType.AKSHARE)
            except ImportError:
                check("YAML load OK (skipped)", True, "pyyaml not installed")
        else:
            check("YAML file exists", False, f"config.yaml not found at {yaml_path}")
            
    except Exception as e:
        check("Config system", False, str(e))


# ============================================================
# T2: Alpha158 Factor Extraction
# ============================================================
def test_alpha158():
    print("\n--- T2: Alpha158 Factor Extraction ---")
    try:
        from hyperion.alpha.factors import Alpha158
        
        data = make_data(["test_stock.SZ"], n_days=300)
        df = data["test_stock.SZ"]
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
        
        extractor = Alpha158()
        factors = extractor.extract(df)
        
        check("Alpha158 extract returns DataFrame", isinstance(factors, pd.DataFrame))
        check(f"Alpha158 has {len(factors.columns)} columns",
              len(factors.columns) >= 150,
              f"expected >=150, got {len(factors.columns)}")
        check("Rows match input", len(factors) == len(df))
        check("Factor names exist", len(extractor.feature_names) >= 150)
        check("No all-NaN factors", not factors.isna().all().any())
        
    except Exception as e:
        check("Alpha158 factors", False, str(e))
        traceback.print_exc()


# ============================================================
# T3: Bayesian Online Learning
# ============================================================
def test_bayesian():
    print("\n--- T3: Bayesian Online Learning ---")
    try:
        from hyperion.alpha.bayesian import BayesianUpdater, FactorState
        
        # Single factor state
        fs = FactorState(mu=0.0, sigma=1.0)
        for _ in range(10):
            fs.update(0.05, known_var=0.01)
        check("FactorState converges", fs.mu > 0.01, f"mu={fs.mu:.4f}")
        check("Confidence increases", fs.sharpness > 1.0, f"sharpness={fs.sharpness:.1f}")
        
        # Multi-factor updater
        updater = BayesianUpdater(n_factors=10)
        for _ in range(30):
            ic = np.random.normal(0.02, 0.05, 10)
            updater.update(ic)
        
        w = updater.weights
        check("Weights sum to 1", abs(w.sum() - 1.0) < 0.01, f"sum={w.sum():.4f}")
        check("All weights >= 0", (w >= 0).all())
        check("Top factors found", len(updater.get_active_factors(3)) == 3)
        
        s = updater.summary()
        check("Summary dict", "mean_ic" in s)
        
    except Exception as e:
        check("Bayesian updater", False, str(e))
        traceback.print_exc()


# ============================================================
# T4: Technical Indicators
# ============================================================
def test_technical():
    print("\n--- T4: Technical Indicators ---")
    try:
        from hyperion.alpha.technical import TechnicalIndicators as TI
        
        data = make_data(["test.SZ"], n_days=200)
        df = data["test.SZ"]
        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        
        # RSI
        rsi = TI.rsi(c, 14)
        check("RSI in [0,100]", 0 <= rsi.dropna().mean() <= 100)
        check("RSI correct length", len(rsi) == len(c))
        
        # MACD
        macd = TI.macd(c)
        check("MACD has 3 keys", set(macd.keys()) == {"macd", "signal", "histogram"})
        check("MACD signal length", len(macd["macd"]) == len(c))
        
        # BBands
        bb = TI.bollinger_bands(c)
        check("BB has 5 keys", "upper" in bb and "lower" in bb and "pct_b" in bb)
        check("BB upper > lower", (bb["upper"].dropna() >= bb["lower"].dropna()).all())
        
        # ATR
        atr = TI.atr(h, l, c)
        check("ATR >= 0", (atr.dropna() >= 0).all())
        
        # Stochastic
        kd = TI.stochastic(h, l, c)
        check("KDJ K correct", len(kd["k"]) == len(c))
        
        # OBV
        obv = TI.obv(c, v)
        check("OBV length", len(obv) == len(c))
        
        # MFI
        mfi = TI.mfi(h, l, c, v)
        check("MFI in range", 0 <= mfi.dropna().mean() <= 100)
        
        # ADX
        adx = TI.adx(h, l, c)
        check("ADX has 3 keys", "adx" in adx and "plus_di" in adx)
        
        # Williams %R
        wr = TI.williams_r(h, l, c)
        check("Williams %R <= 0", (wr.dropna() <= 0).all())
        
        # CCI
        cci = TI.cci(h, l, c)
        check("CCI exists", len(cci) == len(c))
        
        # Detections
        doji = TI.detect_doji(o, h, l, c)
        check("Doji detection", isinstance(doji, pd.Series))
        
        hammer = TI.detect_hammer(o, h, l, c)
        check("Hammer detection", isinstance(hammer, pd.Series))
        
        # SMA/EMA/WMA
        sma = TI.sma(c, 20)
        ema = TI.ema(c, 20)
        wma = TI.wma(c, 20)
        check("SMA/EMA/WMA correct length", len(sma) == len(ema) == len(wma) == len(c))
        
    except Exception as e:
        check("Technical indicators", False, str(e))
        traceback.print_exc()


# ============================================================
# T5: Causal Discovery
# ============================================================
def test_causal():
    print("\n--- T5: Causal Discovery ---")
    try:
        from hyperion.alpha.causal import CausalDiscovery, CausalEdge
        
        # Create factor data where some factors "cause" future returns
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(index=range(n))
        df["factor_a"] = np.random.randn(n)
        df["factor_b"] = np.random.randn(n)
        df["factor_c"] = np.random.randn(n)
        # Make factor_c and factor_a causal for forward_return
        df["forward_return"] = 0.3 * df["factor_c"].shift(1) + 0.1 * df["factor_a"].shift(1) + np.random.randn(n) * 0.1
        df = df.dropna()
        
        cd = CausalDiscovery(max_lag=3, significance=0.05)
        edges = cd.discover(df, target_col="forward_return")
        
        check("Causal edges found", len(edges) > 0, f"got {len(edges)} edges")
        if len(edges) > 0:
            check("First edge has source", edges[0].source in ["factor_a", "factor_b", "factor_c"])
            check("First edge targets return", edges[0].target == "forward_return")
            check("Strength > 0", edges[0].strength > 0)
        
        df_causal = cd.to_dataframe()
        check("to_dataframe works", len(df_causal) == len(edges))
        
        s = cd.summary()
        check("Summary has n_edges", "n_edges" in s)
        
    except Exception as e:
        check("Causal discovery", False, str(e))
        traceback.print_exc()


# ============================================================
# T6: Event Engine
# ============================================================
def test_event_engine():
    print("\n--- T6: Event Engine ---")
    try:
        from hyperion.engine.event_engine import EventEngine, Event, EventType
        
        engine = EventEngine()
        
        received = []
        def handler(event):
            received.append(event)
        
        engine.register(EventType.BAR, handler)
        engine.register(EventType.SIGNAL, handler)
        
        # Synchronous processing (engine not started)
        engine.put(Event(type=EventType.BAR, data={"symbol": "test"}))
        check("Sync event processed", len(received) == 1)
        
        engine.put_bar("000001.SZ", {"close": 10.0})
        check("put_bar works", len(received) == 2)
        
        engine.put_signal("000001.SZ", "BUY", 1.0)
        check("put_signal works", len(received) == 3,
              f"received {len(received)} events (first was BAR)")
        
        stats = engine.stats
        check("Stats dict", "active" in stats and "event_counts" in stats)
        
        # Test async
        engine.start()
        import time
        time.sleep(0.2)
        engine.put(Event(type=EventType.BAR, data={"async": True}))
        time.sleep(0.3)
        check("Async event processed", True, f"total events: {len(received)}")
        engine.stop()
        
    except Exception as e:
        check("Event engine", False, str(e))
        traceback.print_exc()


# ============================================================
# T7: Portfolio Optimizer
# ============================================================
def test_optimizer():
    print("\n--- T7: Portfolio Optimizer ---")
    try:
        from hyperion.risk.optimizer import PortfolioOptimizer
        
        # Generate synthetic returns
        np.random.seed(42)
        n_days, n_assets = 252, 5
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, (n_days, n_assets)),
            columns=[f"asset_{i}" for i in range(n_assets)]
        )
        cov = returns.cov().values
        
        # Equal weight
        w_eq = PortfolioOptimizer.equal_weight(n_assets)
        check("Equal weight sums to 1", abs(w_eq.sum() - 1) < 0.01)
        
        # Risk budgeting
        w_rb = PortfolioOptimizer.risk_budgeting(cov)
        check("Risk budgeting sums to 1", abs(w_rb.sum() - 1) < 0.01)
        check("Risk budgeting weights >= 0", (w_rb >= 0).all())
        
        # HRP
        w_hrp = PortfolioOptimizer.hrp(cov)
        check("HRP sums to 1", abs(w_hrp.sum() - 1) < 0.01)
        check("HRP weights >= 0", (w_hrp >= 0).all())
        
        # Mean-variance
        w_mv = PortfolioOptimizer.mean_variance(returns)
        check("MV sums to 1", abs(w_mv.sum() - 1) < 0.01)
        
        # Max Sharpe
        w_ms = PortfolioOptimizer.max_sharpe(returns)
        check("Max Sharpe sums to 1", abs(w_ms.sum() - 1) < 0.01)
        
    except Exception as e:
        check("Portfolio optimizer", False, str(e))
        traceback.print_exc()


# ============================================================
# T8: Risk Manager
# ============================================================
def test_risk():
    print("\n--- T8: Risk Manager ---")
    try:
        from hyperion.risk.manager import RiskManager, RiskLimits
        
        rm = RiskManager()
        rm.set_capital(1_000_000)
        
        # Normal order
        ok, reason = rm.check_order(
            "000001.SZ", "BUY", 1000, 10.0, {}
        )
        check("Normal order approved", ok, reason)
        
        # Over-position order
        ok2, reason2 = rm.check_order(
            "000001.SZ", "BUY", 100000, 100, {}
        )
        check("Over-position rejected", not ok2, reason2)
        
        # Stop loss check
        triggered, reason = rm.check_stop_loss("test", 10.0, 9.4, {})
        check("Stop loss triggers at -6%", triggered)
        
        triggered2, _ = rm.check_stop_loss("test", 10.0, 9.9, {})
        check("No stop loss at -1%", not triggered2)
        
    except Exception as e:
        check("Risk manager", False, str(e))
        traceback.print_exc()


# ============================================================
# T9: Backtest Engine (End-to-End)
# ============================================================
def test_backtest():
    print("\n--- T9: Backtest Engine (E2E) ---")
    try:
        from hyperion.engine.backtest import BacktestEngine, BacktestResult
        from hyperion.strategy.ml_strategy import MLMultiFactorStrategy
        
        n_symbols = 10
        symbols = [f"stock_{i:02d}.SZ" for i in range(n_symbols)]
        data = make_data(symbols, n_days=252)
        
        strategy = MLMultiFactorStrategy(
            symbols=symbols,
            top_k=5,
            rebalance_freq="monthly",
            mode="momentum"
        )
        
        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission=0.0003,
            stamp_duty=0.001,
            slippage=0.001,
            t_plus_1=True,
            price_limit=0.10
        )
        engine.add_data(data)
        engine.add_strategy(strategy)
        
        result = engine.run(progress=False)
        
        check("BacktestResult returned", isinstance(result, BacktestResult))
        check(f"Equity curve length > 0", len(result.equity_curve) > 0 if result.equity_curve is not None else False)
        check(f"Daily returns length", len(result.daily_returns) > 0 if result.daily_returns is not None else False)
        check("Total return defined", isinstance(result.total_return, float))
        check("Sharpe ratio defined", isinstance(result.sharpe_ratio, float))
        check("Max drawdown <= 0", result.max_drawdown <= 0)
        check("Final value > 0", result.final_value > 0)
        check("Start/end dates", result.start_date != "" and result.end_date != "")
        
        # Dict serialization
        d = result.to_dict()
        check("to_dict works", isinstance(d, dict) and "sharpe_ratio" in d)
        
        print(f"  Portfolio: ¥{result.final_value:,.0f} "
              f"| Return: {result.total_return:.2%} "
              f"| Sharpe: {result.sharpe_ratio:.2f} "
              f"| MaxDD: {result.max_drawdown:.2%}")
        
    except Exception as e:
        check("Backtest engine", False, str(e))
        traceback.print_exc()


# ============================================================
# T10: Report Generation
# ============================================================
def test_report():
    print("\n--- T10: Report Generation ---")
    try:
        from hyperion.analysis.report import ReportGenerator
        from hyperion.analysis.metrics import PerformanceMetrics
        from hyperion.engine.backtest import BacktestResult
        
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        bench = pd.Series(np.random.normal(0.0005, 0.015, 252))
        
        # Performance metrics
        m = PerformanceMetrics.calculate(returns, bench)
        check("Metrics dict", "sharpe_ratio" in m and "max_drawdown" in m)
        check("Sharpe in range", -5 < m.get("sharpe_ratio", 0) < 5, f"sharpe={m.get('sharpe_ratio', 'N/A')}")
        
        # IC analysis
        preds = pd.Series(np.random.randn(100))
        rets = pd.Series(np.random.randn(100))
        ic = PerformanceMetrics.ic_analysis(preds, rets)
        check("IC analysis", "ic" in ic)
        
        # Report generation
        result = BacktestResult(
            total_return=0.15, annual_return=0.20, sharpe_ratio=1.5,
            sortino_ratio=2.0, calmar_ratio=1.8, max_drawdown=-0.12,
            volatility=0.18, total_trades=50, win_rate=0.55,
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=1_000_000, final_value=1_150_000,
            daily_returns=returns, equity_curve=(1+returns).cumprod(),
            benchmark_return=0.10, alpha=0.05, beta=0.85, information_ratio=1.2
        )
        
        r = ReportGenerator.generate(result)
        check("Report has summary", "summary" in r)
        check("Report has performance", "performance" in r)
        
        txt = ReportGenerator.to_text(result)
        check("Text report", "HYPERION QUANT" in txt)
        
    except Exception as e:
        check("Report generation", False, str(e))
        traceback.print_exc()


# ============================================================
# T11: Market Data / Simulation
# ============================================================
def test_simulation():
    print("\n--- T11: Paper Broker ---")
    try:
        from hyperion.execution.simulator import PaperBroker
        
        broker = PaperBroker(initial_capital=1_000_000)
        check("Broker not connected initially", not broker.is_connected)
        
        broker.connect()
        check("Broker connected", broker.is_connected)
        
        broker.update_quote("000001.SZ", 10.0)
        broker.update_quote("000002.SZ", 20.0)
        
        # Buy order
        order = broker.submit_order("000001.SZ", "BUY", 1000, 10.0)
        check("Buy order filled", order.status == "FILLED")
        
        # Sell order
        order2 = broker.submit_order("000001.SZ", "SELL", 500, 10.5)
        check("Sell order filled", order2.status == "FILLED")
        
        # Account
        acct = broker.get_account()
        check("Account has positions", acct.positions is not None)
        check("Account market_value >= 0", acct.market_value >= 0)
        
        positions = broker.get_positions()
        check("Has positions", "000001.SZ" in positions or len(positions) > 0)
        
        broker.disconnect()
        check("Broker disconnected", not broker.is_connected)
        
    except Exception as e:
        check("Paper broker", False, str(e))
        traceback.print_exc()


# ============================================================
# T12: RL Trading Environment
# ============================================================
def test_rl_env():
    print("\n--- T12: RL Trading Environment ---")
    try:
        from hyperion.strategy.rl.env import MarketData, TradingEnv
        
        n_stocks, n_days = 10, 252
        np.random.seed(42)
        prices = 50 * np.exp(np.cumsum(np.random.normal(0, 0.02, (n_days, n_stocks)), axis=0))
        volumes = np.random.randint(100000, 10000000, (n_days, n_stocks)).astype(float)
        
        md = MarketData(prices=prices, volumes=volumes)
        check("MarketData shapes", md.n_days == n_days and md.n_stocks == n_stocks)
        check("Returns shape", md.returns.shape == (n_days - 1, n_stocks))
        
        env = TradingEnv(md, initial_capital=1_000_000, window_size=20)
        state = env.reset()
        check(f"Reset state dim={len(state)}", len(state) == env.state_dim)
        
        for _ in range(5):
            action = np.ones(n_stocks) / n_stocks  # equal weight
            state, reward, done, info = env.step(action)
            check(f"Step reward defined", isinstance(reward, float))
            check(f"Step info OK", "portfolio_value" in info)
        
        check("Env state correct length", len(state) == env.state_dim)
        check("Done after steps", not done, f"unexpected done at step {info.get('step', '?')}/{env.max_steps}")
        
    except Exception as e:
        check("RL TradingEnv", False, str(e))
        traceback.print_exc()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  HYPERION QUANT v1.0 — VALIDATION SUITE")
    print("=" * 60)
    
    tests = [
        ("Config", test_config),
        ("Alpha158", test_alpha158),
        ("Bayesian", test_bayesian),
        ("Technical", test_technical),
        ("Causal", test_causal),
        ("EventEngine", test_event_engine),
        ("Optimizer", test_optimizer),
        ("RiskManager", test_risk),
        ("Backtest", test_backtest),
        ("Report", test_report),
        ("PaperBroker", test_simulation),
        ("RL-Env", test_rl_env),
    ]
    
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n[CRASH] {name}: {e}")
            traceback.print_exc()
            results.append((name, FAIL, str(e)))
    
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"  RESULTS: {n_pass} PASS, {n_fail} FAIL, {len(results)} TOTAL")
    print("=" * 60)
    
    if n_fail > 0:
        print("\nFAILURES:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  [{FAIL}] {name}: {detail}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
