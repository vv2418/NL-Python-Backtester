from __future__ import annotations

from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import StrategySpec

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

_client = OpenAI()

SYSTEM_INSTRUCTIONS = """
You are a trading strategy performance explainer.

Given:
- A JSON-like description of a trading strategy (single asset, long-only).
- A small dictionary of performance metrics (CAGR, max_drawdown, sharpe, num_trades).

Write a concise explanation (<= 200 words) for a retail trader with basic
familiarity with investing. Mention:
- Whether performance was strong or weak overall.
- How risky the strategy was (based on drawdown and Sharpe).
- How frequently it traded (based on num_trades).
- Any obvious caveats (e.g., very few trades, short backtest period).

Do not include code or JSON in your answer. Use plain English.
"""


def summarize_results(
    spec: StrategySpec, metrics: Dict[str, float], model: str = "gpt-4o-mini"
) -> str:
    """Call the LLM to turn metrics into a human explanation."""
    spec_dict = spec.to_dict()
    payload = {
        "strategy_spec": spec_dict,
        "metrics": metrics,
    }

    input_text = (
        "Here is the strategy spec and its performance metrics. "
        "Explain the results clearly but briefly.\n\n"
        f"{payload}"
    )

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": input_text},
        ],
    )
    return response.choices[0].message.content
