from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .strategy_spec import StrategySpec, CrossoverRule, VolFilterRule, Rule


def _evaluate_crossover(rule: CrossoverRule, row_idx: int, df: pd.DataFrame) -> bool:
    fast_col = f"ma_{rule.fast_ma}"
    slow_col = f"ma_{rule.slow_ma}"
    if fast_col not in df.columns or slow_col not in df.columns:
        return False
    # Use .iloc for integer position-based access, ensure scalar values
    row = df.iloc[row_idx]
    fast = row[fast_col]
    slow = row[slow_col]
    # Convert to Python scalar if needed
    if isinstance(fast, pd.Series):
        fast = fast.item() if len(fast) == 1 else fast.iloc[0]
    if isinstance(slow, pd.Series):
        slow = slow.item() if len(slow) == 1 else slow.iloc[0]
    if pd.isna(fast) or pd.isna(slow):
        return False
    if rule.direction == "above":
        return bool(fast > slow)
    else:
        return bool(fast < slow)


def _evaluate_vol_filter(rule: VolFilterRule, row_idx: int, df: pd.DataFrame) -> bool:
    rv_col = f"rv_{rule.window}"
    med_col = f"rv_{rule.window}_med_252"
    if rv_col not in df.columns or med_col not in df.columns:
        return False
    # Use .iloc for integer position-based access, ensure scalar values
    row = df.iloc[row_idx]
    rv = row[rv_col]
    med = row[med_col]
    # Convert to Python scalar if needed
    if isinstance(rv, pd.Series):
        rv = rv.item() if len(rv) == 1 else rv.iloc[0]
    if isinstance(med, pd.Series):
        med = med.item() if len(med) == 1 else med.iloc[0]
    if pd.isna(rv) or pd.isna(med):
        return False
    if rule.relation == "below":
        return bool(rv < med)
    else:
        return bool(rv > med)


def _evaluate_rules(rules: List[Rule], row_idx: int, df: pd.DataFrame) -> bool:
    """Entry rules: all must be true. Exit rules use a different pattern."""
    result = True
    for rule in rules:
        if isinstance(rule, CrossoverRule):
            cond = _evaluate_crossover(rule, row_idx, df)
        elif isinstance(rule, VolFilterRule):
            cond = _evaluate_vol_filter(rule, row_idx, df)
        else:
            cond = False
        result = result and cond
        if not result:
            return False
    return result


def run_backtest(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Simple long-only backtest.

    - Uses close-to-close returns.
    - Entry: when all entry rules are true.
    - Exit: when any exit rule is true.
    - Position changes at the close, impacting next day's return.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    positions = np.zeros(n, dtype=float)

    position = 0.0

    for i in range(n):
        # Entry: all entry rules must be true
        entry_ok = _evaluate_rules(spec.entry_rules, i, df)

        # Exit: any exit rule true -> exit
        exit_ok = False
        for rule in spec.exit_rules:
            if isinstance(rule, CrossoverRule):
                if _evaluate_crossover(rule, i, df):
                    exit_ok = True
                    break
            elif isinstance(rule, VolFilterRule):
                if _evaluate_vol_filter(rule, i, df):
                    exit_ok = True
                    break

        if position == 0.0 and entry_ok:
            position = 1.0
        elif position == 1.0 and exit_ok:
            position = 0.0

        positions[i] = position

    df["position"] = positions

    # Strategy return: use previous day's position on today's return
    df["strategy_return"] = df["position"].shift(1).fillna(0.0) * df["return"]
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()

    return df
