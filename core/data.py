from __future__ import annotations

from typing import Set

import numpy as np
import pandas as pd
import yfinance as yf

from .strategy_spec import StrategySpec, CrossoverRule, VolFilterRule


def load_price_data(spec: StrategySpec) -> pd.DataFrame:
    """Download OHLCV data for the given spec's ticker and date range."""
    df = yf.download(
        spec.ticker,
        start=spec.start_date.isoformat(),
        end=spec.end_date.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No price data returned for {spec.ticker}.")

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["return"] = df["close"].pct_change().fillna(0.0)
    return df


def _required_ma_windows(spec: StrategySpec) -> Set[int]:
    windows: Set[int] = set()
    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, CrossoverRule):
            windows.add(rule.fast_ma)
            windows.add(rule.slow_ma)
    return windows


def _required_vol_windows(spec: StrategySpec) -> Set[int]:
    windows: Set[int] = set()
    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, VolFilterRule):
            windows.add(rule.window)
    return windows


def add_features(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Add all indicators required by the spec (MAs, realized vol, etc.)."""
    df = df.copy()

    # Moving averages
    for w in _required_ma_windows(spec):
        col = f"ma_{w}"
        if col not in df.columns:
            df[col] = df["close"].rolling(window=w, min_periods=1).mean()

    # Realized volatility + 1-year median based on trailing 252 trading days
    for w in _required_vol_windows(spec):
        rv_col = f"rv_{w}"
        med_col = f"rv_{w}_med_252"
        if rv_col not in df.columns:
            df[rv_col] = (
                df["return"].rolling(window=w, min_periods=1).std() * np.sqrt(252.0)
            )
        if med_col not in df.columns:
            df[med_col] = df[rv_col].rolling(window=252, min_periods=1).median()

    return df
