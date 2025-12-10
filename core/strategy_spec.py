from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Literal, Union, Dict, Any


RuleType = Literal["crossover", "vol_filter"]


@dataclass
class BaseRule:
    type: RuleType


@dataclass
class CrossoverRule(BaseRule):
    fast_ma: int
    slow_ma: int
    direction: Literal["above", "below"]  # "above" = fast MA > slow MA
    lookahead_days: int | None = None  # Check condition within next N days (for entry rules)
    duration_days: int | None = None  # Condition must be true for N consecutive days


@dataclass
class VolFilterRule(BaseRule):
    window: int  # e.g. 20-day realized volatility
    threshold: Literal["median_1y"]
    relation: Literal["below", "above"]
    lookahead_days: int | None = None  # Check condition within next N days (for entry rules)
    duration_days: int | None = None  # Condition must be true for N consecutive days


Rule = Union[CrossoverRule, VolFilterRule]


@dataclass
class StrategySpec:
    ticker: str
    start_date: date
    end_date: date
    entry_rules: List[Rule]
    exit_rules: List[Rule]
    metrics: List[str]
    entry_sequential: bool = False  # If True, entry rules must trigger in order (first rule, then subsequent rules within their lookahead windows)

    def to_dict(self) -> Dict[str, Any]:
        def rule_to_dict(rule: Rule) -> Dict[str, Any]:
            base = {"type": rule.type}
            if rule.lookahead_days is not None:
                base["lookahead_days"] = rule.lookahead_days
            if rule.duration_days is not None:
                base["duration_days"] = rule.duration_days
            if isinstance(rule, CrossoverRule):
                base.update(
                    {
                        "fast_ma": rule.fast_ma,
                        "slow_ma": rule.slow_ma,
                        "direction": rule.direction,
                    }
                )
            elif isinstance(rule, VolFilterRule):
                base.update(
                    {
                        "window": rule.window,
                        "threshold": rule.threshold,
                        "relation": rule.relation,
                    }
                )
            return base

        result = {
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "entry_rules": [rule_to_dict(r) for r in self.entry_rules],
            "exit_rules": [rule_to_dict(r) for r in self.exit_rules],
            "metrics": list(self.metrics),
        }
        if self.entry_sequential:
            result["entry_sequential"] = True
        return result


def _parse_rule(rule_dict: Dict[str, Any]) -> Rule:
    rtype = rule_dict.get("type")
    lookahead = rule_dict.get("lookahead_days")
    duration = rule_dict.get("duration_days")
    
    if rtype == "crossover":
        rule = CrossoverRule(
            type="crossover",
            fast_ma=int(rule_dict["fast_ma"]),
            slow_ma=int(rule_dict["slow_ma"]),
            direction=rule_dict["direction"],
            lookahead_days=int(lookahead) if lookahead is not None else None,
            duration_days=int(duration) if duration is not None else None,
        )
        return rule
    elif rtype == "vol_filter":
        rule = VolFilterRule(
            type="vol_filter",
            window=int(rule_dict["window"]),
            threshold=rule_dict.get("threshold", "median_1y"),
            relation=rule_dict.get("relation", "below"),
            lookahead_days=int(lookahead) if lookahead is not None else None,
            duration_days=int(duration) if duration is not None else None,
        )
        return rule
    else:
        raise ValueError(f"Unknown rule type: {rtype}")


def parse_strategy_spec(data: Dict[str, Any]) -> StrategySpec:
    from datetime import date as _date_class

    try:
        start = _date_class.fromisoformat(data["start_date"])
        end = _date_class.fromisoformat(data["end_date"])
    except Exception as e:
        raise ValueError(f"Invalid date format in spec: {e}")

    entry_rules = [_parse_rule(r) for r in data.get("entry_rules", [])]
    exit_rules = [_parse_rule(r) for r in data.get("exit_rules", [])]

    if not entry_rules:
        raise ValueError("At least one entry rule is required.")
    if not exit_rules:
        raise ValueError("At least one exit rule is required.")

    ticker = data["ticker"].upper().strip()
    metrics = data.get("metrics", ["cagr", "max_drawdown", "sharpe"])
    entry_sequential = data.get("entry_sequential", False)

    return StrategySpec(
        ticker=ticker,
        start_date=start,
        end_date=end,
        entry_rules=entry_rules,
        exit_rules=exit_rules,
        metrics=metrics,
        entry_sequential=entry_sequential,
    )
