from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import parse_strategy_spec, StrategySpec
from utils.metrics_tracker import log_metrics

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Initialize a single shared client (expects OPENAI_API_KEY in env)
_client = OpenAI()

SYSTEM_INSTRUCTIONS = """
You are a trading strategy specification generator.

Your job is to read a natural-language description of a single-asset,
long-only backtest and convert it into a JSON object with this schema:

{
  "ticker": "AAPL",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "entry_rules": [
    {
      "type": "crossover",
      "fast_ma": 10,
      "slow_ma": 50,
      "direction": "above",
      "lookahead_days": 3
    },
    {
      "type": "vol_filter",
      "window": 20,
      "threshold": "median_1y",
      "relation": "below"
    }
  ],
  "exit_rules": [
    {
      "type": "crossover",
      "fast_ma": 10,
      "slow_ma": 50,
      "direction": "below",
      "duration_days": 2
    }
  ],
  "entry_sequential": true,
  "metrics": ["cagr", "max_drawdown", "sharpe"]
}

Rules you MUST follow:
- Only support one ticker symbol (like AAPL, SPY, TSLA).
- Only support long-only strategies (no shorting, no leverage).
- Only support these rule types:
  - Moving-average crossover rules.
  - Volatility filters comparing realized vol to its 1-year median.
- If the user asks for unsupported features (options, multi-asset portfolios,
  intraday data, complex position sizing), ignore those unsupported parts and
  create the closest approximation you can with the supported rule set,
  but do not change any numeric parameters or dates that the user explicitly
  specified.
- Use ISO 8601 dates (YYYY-MM-DD).
- Always include at least one entry rule and one exit rule.
- For metrics, default to ["cagr", "max_drawdown", "sharpe"] if the user does
  not specify metrics.
- Do NOT fix, correct, or normalize user-provided parameters or dates.
  If the user gives strange or unrealistic values (such as very small or very
  large moving-average windows, or a start_date that is after end_date), only then ask for confirmation.
- If the user describes logically conflicting conditions (for example, the same
  moving averages being both "above" and "below" each other at the same time),
  ask them whether they want to represent them as multiple rules in the JSON, or if they want to use a single rule.
- Do not add rules the user did not ask for, and do not delete rules that the
  user did ask for.

TEMPORAL CONSTRAINTS (CRITICAL - READ CAREFULLY):
You MUST use these fields when the user's prompt contains temporal language. This is NOT optional.

1. "lookahead_days" (optional integer): REQUIRED when user says:
   - "within the next N trading days"
   - "within N days"
   - "in the next N days"
   - "within N trading days"
   Add this field to the rule that must happen within the time window.
   Example: "volatility drops below median within 3 days" → vol_filter rule gets "lookahead_days": 3

2. "duration_days" (optional integer): REQUIRED when user says:
   - "for N consecutive days"
   - "for N straight days"
   - "has been [condition] for N days"
   - "N days in a row"
   Add this field to the rule that must persist for multiple days.
   Example: "MA below for 2 straight days" → crossover rule gets "duration_days": 2

3. "entry_sequential" (optional boolean): REQUIRED ONLY when user describes SEQUENTIAL ENTRY LOGIC:
   - "first [condition A], then [condition B]" (in entry rules)
   - "if first [A], then [B] within N days" (in entry rules)
   - "first [A] happens, then [B]" (in entry rules)
   Set "entry_sequential": true at the top level of the JSON ONLY when entry rules are sequential.
   When sequential: first entry rule triggers, then subsequent rules must trigger within their lookahead_days windows.
   Example: "Enter if first MA crosses above, then volatility drops within 3 days" → 
     entry_sequential: true, first rule (crossover) has no lookahead, second rule (vol_filter) has lookahead_days: 3

CRITICAL: "entry_sequential" applies ONLY to entry rules. If temporal constraints (lookahead_days, duration_days) appear ONLY in exit rules, do NOT set entry_sequential to true. Only set entry_sequential: true when the user explicitly describes sequential logic for entry conditions (multiple entry rules that must trigger in order).

IMPORTANT: If the user's prompt contains ANY of the phrases above, you MUST include the corresponding temporal constraint fields. Do not ignore temporal language in the prompt.

You MUST output a single JSON object with no explanation or commentary.
"""


USER_TEMPLATE = 'User strategy description:\n\n"""{user_text}"""\n'


def translate_to_spec(user_text: str, model: str = "gpt-4o-mini") -> StrategySpec:
    prompt = USER_TEMPLATE.format(user_text=user_text)
    
    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        # Extract token usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        json_str = response.choices[0].message.content
        data: Dict[str, Any] = json.loads(json_str)
        spec = parse_strategy_spec(data)
        success = True
        
        return spec
    except Exception as e:
        error_msg = str(e)
        input_tokens = 0
        output_tokens = 0
        raise
    finally:
        latency = time.time() - start_time
        log_metrics(
            task_type="translation",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            success=success,
            error_message=error_msg,
        )
