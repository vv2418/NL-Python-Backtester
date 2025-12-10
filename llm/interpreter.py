from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import StrategySpec
from utils.metrics_tracker import log_metrics

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Initialize a single shared client (expects OPENAI_API_KEY in env)
_client = OpenAI()

SYSTEM_INSTRUCTIONS = """
You are a trading strategy interpretation explainer. Your job is to explain how a natural language strategy description was interpreted into a structured trading strategy specification.

Given:
1. The user's original natural language description
2. The parsed strategy specification (JSON format)

Provide a clear, concise explanation that:
1. Summarizes how the strategy was interpreted (ticker, dates, entry/exit rules, metrics)
2. Identifies ONLY critical ambiguities that would prevent execution or cause errors

CRITICAL RULES:
- Do NOT ask about optional features the user didn't mention (position sizing, stop-losses, etc.). If they wanted these, they would have included them.
- Do NOT mention assumptions about defaults that are already handled (e.g., SMA type, execution method). These are standard and don't need confirmation.
- Only mention ambiguities if they would cause the backtest to fail or produce incorrect results.
- Be brief. If everything is clear and executable, just summarize the interpretation.

Be friendly and clear. Use plain English, avoid jargon.
"""


def explain_interpretation(
    user_text: str, spec: StrategySpec, model: str = "gpt-4o-mini"
) -> str:
    """Generate an explanation of how the user's strategy was interpreted."""
    spec_dict = spec.to_dict()
    spec_json = json.dumps(spec_dict, indent=2)
    
    prompt = f"""Original user description:

"{user_text}"

Parsed strategy specification (JSON):

```json
{spec_json}
```

Provide a clear, concise explanation with these sections:

1. **How I interpreted this strategy** - Summarize the key elements: ticker, date range, entry rules, exit rules, and metrics.

2. **Critical ambiguities** (ONLY if they would prevent execution or cause errors):
   - Only mention ambiguities that would make the backtest fail or produce incorrect results
   - Do NOT ask about optional features (position sizing, stop-losses, order types, etc.) - if the user wanted these, they would have mentioned them
   - Do NOT mention standard defaults (SMA type, close-to-close execution, etc.) - these are handled automatically
   - If there are no critical ambiguities, omit this section entirely

Be brief and direct. If the strategy is clear and executable, focus on summarizing the interpretation."""

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
            task_type="interpretation",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            success=success,
            error_message=error_msg,
        )

