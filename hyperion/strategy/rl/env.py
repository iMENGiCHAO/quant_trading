"""
TradingEnv - FinRL-inspired Gym Environment for A-Share
=========================================================
State:  portfolio weights + market features
Action: continuous portfolio weights (sum to 1)
Reward: risk-adjusted return (Sharpe-like)

A-share constraints:
- T+1 settlement
- +/-10% daily price limit
- ST stock exclusion

Pure numpy implementation, compatible with gym interface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import gym
    from gym import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False

logger = logging.getLogger(__name__)


class MarketData:
    """历史行情数据容器"""
    
    def __init__(self, prices: np.ndarray, volumes: np.ndarray,
                 st_flags: Optional[np.ndarray] = None):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.volumes = np.asarray(volumes, dtype=np.float64)
        self.n_days, self.n_stocks = self.prices.shape
        
        if st_flags is None:
            self.st_flags = np.zeros_like(prices, dtype=bool)
        else:
            self.st_flags = np.asarray(st_flags, dtype=bool)
    
    @property
    def returns(self) -> np.ndarray:
        """日收益率矩阵 (n_days-1, n_stocks)"""
        return np.diff(self.prices, axis=0) / (self.prices[:-1] + 1e-12)


class TradingEnv:
    """FinRL风格交易环境
    
    Usage:
        env = TradingEnv(market_data, initial_capital=1_000_000)
        state = env.reset()
        for step in range(env.max_steps):
            action = agent.act(state)  # portfolio weights
            state, reward, done, info = env.step(action)
    """
    
    def __init__(self, market_data: MarketData,
                 initial_capital: float = 1_000_000.0,
                 transaction_cost_pct: float = 0.001,
                 window_size: int = 20,
                 max_positions: int = 30):
        self.data = market_data
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.max_positions = max_positions
        
        self.n_stocks = market_data.n_stocks
        self.max_steps = market_data.n_days - window_size - 1
        
        # State spaces
        self.state_dim = self.n_stocks * 3 + 3  # weights + returns + volume_chg + cash + step
        
        if _HAS_GYM:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float64
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float64
            )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self._step = 0
        self.capital = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        # 持仓: (n_stocks,) shares
        self.holdings = np.zeros(self.n_stocks)
        self.holdings_value = np.zeros(self.n_stocks)
        
        # T+1: 今日买入的不能卖出
        self._bought_today = np.zeros(self.n_stocks, dtype=bool)
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步交易
        
        Args:
            action: target portfolio weights (n_stocks,), sum ≈ 1.0
            
        Returns:
            state, reward, done, info
        """
        action = np.asarray(action, dtype=np.float64)
        
        # 1. 获取当前价格
        current_idx = self._step + self.window_size
        prices = self.data.prices[current_idx]
        prev_prices = self.data.prices[current_idx - 1]
        
        # 2. 计算当前持仓价值
        self.holdings_value = self.holdings * prices
        current_portfolio = self.capital + np.sum(self.holdings_value)
        
        # 3. 计算目标持仓
        # 归一化 (去负值, T+1约束)
        target_weights = np.maximum(action, 0)
        w_sum = target_weights.sum()
        if w_sum > 0:
            target_weights = target_weights / w_sum
        
        # 应用价格限制 (涨跌停无法交易)
        price_change = prices / (prev_prices + 1e-12) - 1
        can_trade = np.abs(price_change) < 0.098  # < 9.8% (留缓冲)
        
        # ST过滤
        if hasattr(self.data, 'st_flags'):
            st_today = self.data.st_flags[current_idx]
            can_trade = can_trade & (~st_today)
        
        target_value = target_weights * current_portfolio * can_trade
        
        # 4. 计算交易
        trade_value = target_value - self.holdings_value
        
        # T+1: 不能卖出今日买入的
        sell_mask = trade_value < 0
        trade_value[sell_mask & self._bought_today] = 0
        
        # 交易成本
        cost = np.sum(np.abs(trade_value)) * self.transaction_cost_pct
        
        # 5. 更新持仓
        self.holdings += trade_value / (prices + 1e-12)
        self.capital -= (np.sum(trade_value) + cost)
        self.holdings_value = self.holdings * prices
        
        # T+1 tracking
        self._bought_today = trade_value > 0
        
        # 6. 计算新组合价值
        new_portfolio = self.capital + np.sum(self.holdings_value)
        self.portfolio_value = new_portfolio
        
        # 7. 奖励 (对数收益率)
        reward = np.log(new_portfolio / (current_portfolio + 1e-12))
        
        # 8. 状态更新
        self._step += 1
        done = self._step >= self.max_steps
        
        state = self._get_state()
        info = {
            "portfolio_value": float(new_portfolio),
            "cash": float(self.capital),
            "cost": float(cost),
            "return": float(new_portfolio / current_portfolio - 1),
            "positions": int(np.sum(self.holdings > 0)),
            "step": self._step
        }
        
        return state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """构建当前状态向量"""
        current_idx = self._step + self.window_size
        
        # 持仓权重
        total = self.capital + np.sum(self.holdings_value) + 1e-12
        weights = self.holdings_value / total
        
        # 近期收益率 (window_size天)
        start = max(0, current_idx - self.window_size)
        prices_window = self.data.prices[start:current_idx + 1]
        if len(prices_window) > 1:
            recent_returns = (prices_window[-1] / (prices_window[0] + 1e-12) - 1)
        else:
            recent_returns = np.zeros(self.n_stocks)
        
        # 成交量变化
        if current_idx > 0:
            vol_change = self.data.volumes[current_idx] / (self.data.volumes[current_idx - 1] + 1e-12) - 1
        else:
            vol_change = np.zeros(self.n_stocks)
        
        # 全局特征
        global_feats = np.array([
            self.capital / total,
            self._step / max(1, self.max_steps),
            np.sum(weights > 0) / max(1, self.n_stocks)
        ])
        
        return np.concatenate([weights, recent_returns, vol_change, global_feats])
    
    def render(self, mode: str = "human"):
        """渲染当前状态"""
        print(f"Step: {self._step}/{self.max_steps} | "
              f"Portfolio: {self.portfolio_value:,.0f} | "
              f"Cash: {self.capital:,.0f} | "
              f"Positions: {int(np.sum(self.holdings > 0))}")
