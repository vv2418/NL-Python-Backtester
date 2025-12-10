from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .strategy_spec import StrategySpec, CrossoverRule, VolFilterRule, Rule


@dataclass
class Trade:
    """Represents a single completed trade."""
    entry_date: str
    entry_price: float
    entry_reason: str
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None


def _to_scalar(val):
    """Convert a pandas Series or numpy array to a Python scalar."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0]) if len(val) > 0 else 0.0
    elif hasattr(val, 'item'):
        return val.item()
    return float(val)


def _format_crossover_reason(rule: CrossoverRule, row: pd.Series, is_entry: bool) -> str:
    """Format a human-readable reason for a crossover rule trigger."""
    fast_col = f"ma_{rule.fast_ma}"
    slow_col = f"ma_{rule.slow_ma}"
    fast_val = _to_scalar(row[fast_col])
    slow_val = _to_scalar(row[slow_col])
    action = "Entry" if is_entry else "Exit"
    direction = "above" if rule.direction == "above" else "below"
    return f"{action}: {rule.fast_ma}-day MA ({fast_val:.2f}) crossed {direction} {rule.slow_ma}-day MA ({slow_val:.2f})"


def _format_vol_filter_reason(rule: VolFilterRule, row: pd.Series, is_entry: bool) -> str:
    """Format a human-readable reason for a volatility filter trigger."""
    rv_col = f"rv_{rule.window}"
    med_col = f"rv_{rule.window}_med_252"
    rv_val = _to_scalar(row[rv_col])
    med_val = _to_scalar(row[med_col])
    action = "Entry" if is_entry else "Exit"
    relation = "below" if rule.relation == "below" else "above"
    return f"{action}: {rule.window}-day RV ({rv_val:.2%}) {relation} 1Y median ({med_val:.2%})"


def _get_triggered_rules(
    rules: List[Rule], row_idx: int, df: pd.DataFrame, is_entry: bool
) -> Tuple[bool, List[str]]:
    """Evaluate rules and return (all_passed, list of triggered rule reasons)."""
    row = df.iloc[row_idx]
    reasons = []
    all_passed = True

    for rule in rules:
        if isinstance(rule, CrossoverRule):
            passed = _evaluate_crossover(rule, row_idx, df)
            if passed:
                reasons.append(_format_crossover_reason(rule, row, is_entry))
        elif isinstance(rule, VolFilterRule):
            passed = _evaluate_vol_filter(rule, row_idx, df)
            if passed:
                reasons.append(_format_vol_filter_reason(rule, row, is_entry))
        else:
            passed = False

        if not passed:
            all_passed = False

    return all_passed, reasons


def _evaluate_crossover(rule: CrossoverRule, row_idx: int, df: pd.DataFrame) -> bool:
    """Evaluate crossover rule, supporting lookahead and duration constraints."""
    fast_col = f"ma_{rule.fast_ma}"
    slow_col = f"ma_{rule.slow_ma}"
    if fast_col not in df.columns or slow_col not in df.columns:
        return False
    
    n = len(df)
    
    # Handle duration requirement: condition must be true for N consecutive days
    if rule.duration_days is not None:
        duration = rule.duration_days
        if row_idx < duration - 1:
            return False
        for offset in range(duration):
            check_idx = row_idx - offset
            if check_idx < 0:
                return False
            row = df.iloc[check_idx]
            fast = row[fast_col]
            slow = row[slow_col]
            if isinstance(fast, pd.Series):
                fast = fast.item() if len(fast) == 1 else fast.iloc[0]
            if isinstance(slow, pd.Series):
                slow = slow.item() if len(slow) == 1 else slow.iloc[0]
            if pd.isna(fast) or pd.isna(slow):
                return False
            condition_met = (fast > slow) if rule.direction == "above" else (fast < slow)
            if not condition_met:
                return False
        return True
    
    # Handle lookahead: check if condition is true within next N days
    if rule.lookahead_days is not None:
        lookahead = rule.lookahead_days
        for offset in range(lookahead + 1):
            check_idx = row_idx + offset
            if check_idx >= n:
                continue
            row = df.iloc[check_idx]
            fast = row[fast_col]
            slow = row[slow_col]
            if isinstance(fast, pd.Series):
                fast = fast.item() if len(fast) == 1 else fast.iloc[0]
            if isinstance(slow, pd.Series):
                slow = slow.item() if len(slow) == 1 else slow.iloc[0]
            if pd.isna(fast) or pd.isna(slow):
                continue
            condition_met = (fast > slow) if rule.direction == "above" else (fast < slow)
            if condition_met:
                return True
        return False
    
    # Standard evaluation: check current row only
    row = df.iloc[row_idx]
    fast = row[fast_col]
    slow = row[slow_col]
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
    """Evaluate volatility filter rule, supporting lookahead and duration constraints."""
    rv_col = f"rv_{rule.window}"
    med_col = f"rv_{rule.window}_med_252"
    if rv_col not in df.columns or med_col not in df.columns:
        return False
    
    n = len(df)
    
    # Handle duration requirement: condition must be true for N consecutive days
    if rule.duration_days is not None:
        duration = rule.duration_days
        if row_idx < duration - 1:
            return False
        for offset in range(duration):
            check_idx = row_idx - offset
            if check_idx < 0:
                return False
            row = df.iloc[check_idx]
            rv = row[rv_col]
            med = row[med_col]
            if isinstance(rv, pd.Series):
                rv = rv.item() if len(rv) == 1 else rv.iloc[0]
            if isinstance(med, pd.Series):
                med = med.item() if len(med) == 1 else med.iloc[0]
            if pd.isna(rv) or pd.isna(med):
                return False
            condition_met = (rv < med) if rule.relation == "below" else (rv > med)
            if not condition_met:
                return False
        return True
    
    # Handle lookahead: check if condition is true within next N days
    if rule.lookahead_days is not None:
        lookahead = rule.lookahead_days
        for offset in range(lookahead + 1):
            check_idx = row_idx + offset
            if check_idx >= n:
                continue
            row = df.iloc[check_idx]
            rv = row[rv_col]
            med = row[med_col]
            if isinstance(rv, pd.Series):
                rv = rv.item() if len(rv) == 1 else rv.iloc[0]
            if isinstance(med, pd.Series):
                med = med.item() if len(med) == 1 else med.iloc[0]
            if pd.isna(rv) or pd.isna(med):
                continue
            condition_met = (rv < med) if rule.relation == "below" else (rv > med)
            if condition_met:
                return True
        return False
    
    # Standard evaluation: check current row only
    row = df.iloc[row_idx]
    rv = row[rv_col]
    med = row[med_col]
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


def _evaluate_rule_at_index(rule: Rule, row_idx: int, df: pd.DataFrame) -> bool:
    """Evaluate a rule at a specific index, ignoring lookahead/duration constraints."""
    if isinstance(rule, CrossoverRule):
        # Temporarily disable lookahead and duration to check specific day
        original_lookahead = rule.lookahead_days
        original_duration = rule.duration_days
        rule.lookahead_days = None
        rule.duration_days = None
        result = _evaluate_crossover(rule, row_idx, df)
        rule.lookahead_days = original_lookahead
        rule.duration_days = original_duration
        return result
    elif isinstance(rule, VolFilterRule):
        original_lookahead = rule.lookahead_days
        original_duration = rule.duration_days
        rule.lookahead_days = None
        rule.duration_days = None
        result = _evaluate_vol_filter(rule, row_idx, df)
        rule.lookahead_days = original_lookahead
        rule.duration_days = original_duration
        return result
    return False


def _evaluate_sequential_entry(rules: List[Rule], row_idx: int, df: pd.DataFrame) -> bool:
    """Evaluate sequential entry rules: first rule must trigger, then subsequent rules within their lookahead windows.
    
    Returns True if:
    - First rule triggers at row_idx (ignoring its lookahead if present)
    - Each subsequent rule triggers within its lookahead_days window (starting from row_idx)
    - If a rule has no lookahead_days, it must trigger at the same time as the first rule
    """
    if not rules:
        return False
    
    n = len(df)
    
    # First rule must trigger at current row (check without lookahead)
    first_rule = rules[0]
    if not _evaluate_rule_at_index(first_rule, row_idx, df):
        return False
    
    # For subsequent rules, check if they trigger within their lookahead windows
    for rule in rules[1:]:
        if rule.lookahead_days is None:
            # No lookahead: must trigger at same time as first rule
            if not _evaluate_rule_at_index(rule, row_idx, df):
                return False
        else:
            # Has lookahead: check if it triggers within the window (including same day)
            lookahead = rule.lookahead_days
            rule_triggered = False
            for offset in range(lookahead + 1):  # Include same day (offset 0)
                check_idx = row_idx + offset
                if check_idx >= n:
                    break
                if _evaluate_rule_at_index(rule, check_idx, df):
                    rule_triggered = True
                    break
            if not rule_triggered:
                return False
    
    return True


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
        # Entry: use sequential evaluation if enabled, otherwise standard evaluation
        if spec.entry_sequential:
            entry_ok = _evaluate_sequential_entry(spec.entry_rules, i, df)
        else:
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


def extract_trades(df: pd.DataFrame, spec: StrategySpec) -> List[Trade]:
    """Extract detailed trade log from backtest results.

    Returns a list of Trade objects with entry/exit dates, prices, and reasons.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    position = 0.0

    for i in range(n):
        row = df.iloc[i]
        # Handle date - could be a Series with MultiIndex from yfinance
        if "date" in df.columns:
            date_val = df["date"].iloc[i]
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]
        else:
            date_str = str(i)
        # Handle close price - ensure scalar
        close_price = float(df["close"].iloc[i])

        # Check entry conditions
        if spec.entry_sequential:
            entry_ok = _evaluate_sequential_entry(spec.entry_rules, i, df)
            # For sequential, collect reasons - first rule at current day, subsequent rules within their windows
            entry_reasons = []
            if entry_ok:
                # First rule triggered at current day
                first_rule = spec.entry_rules[0]
                if isinstance(first_rule, CrossoverRule):
                    entry_reasons.append(_format_crossover_reason(first_rule, row, is_entry=True))
                elif isinstance(first_rule, VolFilterRule):
                    entry_reasons.append(_format_vol_filter_reason(first_rule, row, is_entry=True))
                
                # Find when subsequent rules triggered within their lookahead windows
                for rule in spec.entry_rules[1:]:
                    found = False
                    lookahead = rule.lookahead_days if rule.lookahead_days is not None else 0
                    for offset in range(lookahead + 1):
                        check_idx = i + offset
                        if check_idx < len(df) and _evaluate_rule_at_index(rule, check_idx, df):
                            check_row = df.iloc[check_idx]
                            if isinstance(rule, CrossoverRule):
                                entry_reasons.append(_format_crossover_reason(rule, check_row, is_entry=True))
                            elif isinstance(rule, VolFilterRule):
                                entry_reasons.append(_format_vol_filter_reason(rule, check_row, is_entry=True))
                            found = True
                            break
                    if not found and rule.lookahead_days is None:
                        # No lookahead, should have triggered at same time
                        if isinstance(rule, CrossoverRule):
                            entry_reasons.append(_format_crossover_reason(rule, row, is_entry=True))
                        elif isinstance(rule, VolFilterRule):
                            entry_reasons.append(_format_vol_filter_reason(rule, row, is_entry=True))
        else:
            entry_ok, entry_reasons = _get_triggered_rules(spec.entry_rules, i, df, is_entry=True)

        # Check exit conditions (any exit rule)
        exit_reasons = []
        exit_ok = False
        for rule in spec.exit_rules:
            if isinstance(rule, CrossoverRule):
                if _evaluate_crossover(rule, i, df):
                    exit_reasons.append(_format_crossover_reason(rule, row, is_entry=False))
                    exit_ok = True
                    break
            elif isinstance(rule, VolFilterRule):
                if _evaluate_vol_filter(rule, i, df):
                    exit_reasons.append(_format_vol_filter_reason(rule, row, is_entry=False))
                    exit_ok = True
                    break

        # Handle state transitions
        if position == 0.0 and entry_ok:
            # Enter new trade
            current_trade = Trade(
                entry_date=date_str,
                entry_price=close_price,
                entry_reason=" | ".join(entry_reasons) if entry_reasons else "All entry rules satisfied",
            )
            position = 1.0

        elif position == 1.0 and exit_ok and current_trade is not None:
            # Exit current trade
            current_trade.exit_date = date_str
            current_trade.exit_price = close_price
            current_trade.exit_reason = " | ".join(exit_reasons) if exit_reasons else "Exit rule triggered"
            current_trade.pnl_pct = (close_price / current_trade.entry_price - 1) * 100
            trades.append(current_trade)
            current_trade = None
            position = 0.0

    # Handle open trade at end of backtest
    if current_trade is not None and position == 1.0:
        if "date" in df.columns:
            date_val = df["date"].iloc[-1]
            if hasattr(date_val, 'strftime'):
                exit_date_str = date_val.strftime('%Y-%m-%d')
            else:
                exit_date_str = str(date_val)[:10]
        else:
            exit_date_str = "End"
        current_trade.exit_date = exit_date_str
        current_trade.exit_price = float(df["close"].iloc[-1])
        current_trade.exit_reason = "End of backtest period (still holding)"
        current_trade.pnl_pct = (current_trade.exit_price / current_trade.entry_price - 1) * 100
        trades.append(current_trade)

    return trades


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert list of trades to a pandas DataFrame for display."""
    if not trades:
        return pd.DataFrame(columns=[
            "Entry Date", "Entry Price", "Entry Reason",
            "Exit Date", "Exit Price", "Exit Reason", "P&L %"
        ])

    data = []
    for t in trades:
        data.append({
            "Entry Date": t.entry_date,
            "Entry Price": f"${t.entry_price:.2f}",
            "Entry Reason": t.entry_reason,
            "Exit Date": t.exit_date or "Open",
            "Exit Price": f"${t.exit_price:.2f}" if t.exit_price else "N/A",
            "Exit Reason": t.exit_reason or "N/A",
            "P&L %": f"{t.pnl_pct:+.2f}%" if t.pnl_pct is not None else "N/A",
        })

    return pd.DataFrame(data)
