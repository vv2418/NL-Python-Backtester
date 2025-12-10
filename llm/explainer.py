from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import StrategySpec
from utils.metrics_tracker import log_metrics

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

    start_time = time.time()
    success = False
    error_msg = None
    
    try:
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": input_text},
            ],
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        result = response.choices[0].message.content
        success = True
        return result
    except Exception as e:
        error_msg = str(e)
        input_tokens = 0
        output_tokens = 0
        raise
    finally:
        latency = time.time() - start_time
        log_metrics(
            task_type="explanation",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            success=success,
            error_message=error_msg,
        )
