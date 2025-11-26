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
  intraday data, complex position sizing), ignore those parts and create the
  closest approximation you can with the supported rule set.
- Use ISO 8601 dates (YYYY-MM-DD).
- Always include at least one entry rule and one exit rule.
- For metrics, default to ["cagr", "max_drawdown", "sharpe"] if the user does
  not specify metrics.

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
