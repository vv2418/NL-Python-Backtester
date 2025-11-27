from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import parse_strategy_spec, StrategySpec

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
      "direction": "above"
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
      "direction": "below"
    }
  ],
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
  large moving-average windows, or a start_date that is after end_date), encode
  them exactly as written in the JSON.
- If the user describes logically conflicting conditions (for example, the same
  moving averages being both "above" and "below" each other at the same time),
  represent them as multiple rules in the JSON instead of resolving or
  simplifying the conflict.
- Do not add rules the user did not ask for, and do not delete rules that the
  user did ask for.

You MUST output a single JSON object with no explanation or commentary.
"""


USER_TEMPLATE = 'User strategy description:\n\n"""{user_text}"""\n'


def translate_to_spec(user_text: str, model: str = "gpt-4o-mini") -> StrategySpec:
    prompt = USER_TEMPLATE.format(user_text=user_text)

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    json_str = response.choices[0].message.content
    data: Dict[str, Any] = json.loads(json_str)
    return parse_strategy_spec(data)
