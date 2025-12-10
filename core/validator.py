from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .strategy_spec import StrategySpec, CrossoverRule, VolFilterRule, Rule
from .backtester import _evaluate_crossover, _evaluate_vol_filter, _evaluate_sequential_entry


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def validate_spec(spec: StrategySpec) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    if spec.start_date >= spec.end_date:
        errors.append("Start date must be before end date.")

    if not spec.entry_rules:
        errors.append("At least one entry rule is required.")
    if not spec.exit_rules:
        errors.append("At least one exit rule is required.")

    if not spec.metrics:
        warnings.append("No metrics specified; default metrics will be used.")

    crossover_entry = {}
    vol_entry = {}

    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, CrossoverRule):
            if rule.fast_ma <= 0 or rule.slow_ma <= 0:
                errors.append("Moving average windows must be positive integers.")
            if rule.fast_ma == rule.slow_ma:
                errors.append("Fast and slow moving averages must differ.")
            if rule.fast_ma < 5 or rule.slow_ma < 5:
                warnings.append("Very small moving average windows (under 5 days) may be unstable or overly reactive.")
            if rule.fast_ma > 200 or rule.slow_ma > 200:
                warnings.append("Very large moving average windows (over 200 days) may make the strategy slow and unresponsive.")

            key = (rule.fast_ma, rule.slow_ma)
            if key not in crossover_entry:
                crossover_entry[key] = set()
            if rule in spec.entry_rules:
                crossover_entry[key].add(rule.direction)
        elif isinstance(rule, VolFilterRule):
            if rule.window <= 1:
                errors.append("Volatility window must be greater than 1.")
            if rule.window > 252 * 5:
                warnings.append("Very large volatility windows may dilute signal responsiveness.")
            key = (rule.window, rule.threshold)
            if key not in vol_entry:
                vol_entry[key] = set()
            if rule in spec.entry_rules:
                vol_entry[key].add(rule.relation)

    for key, directions in crossover_entry.items():
        if "above" in directions and "below" in directions:
            errors.append("Entry rules require the same moving averages to be both above and below each other, which is impossible.")

    for key, relations in vol_entry.items():
        if "above" in relations and "below" in relations:
            errors.append("Entry rules require volatility to be both above and below the same threshold, which is impossible.")

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings)


def _evaluate_entry_rules(spec: StrategySpec, row_idx: int, df: pd.DataFrame) -> bool:
    if spec.entry_sequential:
        return _evaluate_sequential_entry(spec.entry_rules, row_idx, df)
    result = True
    for rule in spec.entry_rules:
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


def _evaluate_exit_any(spec: StrategySpec, row_idx: int, df: pd.DataFrame) -> bool:
    for rule in spec.exit_rules:
        if isinstance(rule, CrossoverRule):
            if _evaluate_crossover(rule, row_idx, df):
                return True
        elif isinstance(rule, VolFilterRule):
            if _evaluate_vol_filter(rule, row_idx, df):
                return True
    return False


def validate_with_data(spec: StrategySpec, df: pd.DataFrame) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("No price data is available for the requested period.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    n = len(df)
    max_ma = 0
    max_vol = 0
    max_lookahead = 0
    max_duration = 0

    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, CrossoverRule):
            max_ma = max(max_ma, rule.fast_ma, rule.slow_ma)
        elif isinstance(rule, VolFilterRule):
            max_vol = max(max_vol, rule.window)
        if rule.lookahead_days is not None:
            max_lookahead = max(max_lookahead, rule.lookahead_days)
        if rule.duration_days is not None:
            max_duration = max(max_duration, rule.duration_days)

    required_len = 0
    if max_ma > 0:
        required_len = max(required_len, max_ma + 10)
    if max_vol > 0:
        required_len = max(required_len, max_vol + 252)
    if max_lookahead > 0:
        required_len = max(required_len, max_lookahead + 10)
    if max_duration > 0:
        required_len = max(required_len, max_duration + 10)

    if required_len > 0 and n < required_len:
        warnings.append(
            f"Strategy uses long lookback windows (up to {required_len} days) but only {n} data points are available. Early signal values may be unreliable."
        )

    any_entry = False
    any_exit = False

    for i in range(n):
        if _evaluate_entry_rules(spec, i, df):
            any_entry = True
        if _evaluate_exit_any(spec, i, df):
            any_exit = True
        if any_entry and any_exit:
            break

    if not any_entry:
        warnings.append("Given the historical data and rules, this strategy is unlikely to generate any entries. It may produce zero trades.")
    if any_entry and not any_exit:
        warnings.append("Entry conditions can occur, but exit conditions never trigger on this data. Positions may never close once opened.")

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings)
