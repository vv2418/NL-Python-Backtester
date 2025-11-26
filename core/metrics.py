from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_cagr(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
    if equity_curve.empty:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    n_days = len(equity_curve)
    years = n_days / trading_days_per_year
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    cum_max = equity_curve.cummax()
    drawdowns = equity_curve / cum_max - 1.0
    return float(drawdowns.min())


def compute_sharpe(daily_returns: pd.Series, trading_days_per_year: int = 252) -> float:
    if daily_returns.empty:
        return 0.0
    mean = daily_returns.mean()
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return float((mean / std) * (trading_days_per_year ** 0.5))


def compute_basic_metrics(df: pd.DataFrame) -> Dict[str, float]:
    equity = df["equity_curve"]
    strat_ret = df["strategy_return"]
    metrics = {
        "cagr": compute_cagr(equity),
        "max_drawdown": compute_max_drawdown(equity),
        "sharpe": compute_sharpe(strat_ret),
        "num_trades": float(
            ((df["position"].diff().abs() > 0) & (df["position"] == 1)).sum()
        ),
    }
    return metrics
