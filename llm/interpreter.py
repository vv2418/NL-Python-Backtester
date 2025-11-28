from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from core.strategy_spec import StrategySpec

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
1. Summarizes how the strategy was interpreted
2. Highlights any assumptions made (e.g., default values, implicit rules)
3. Identifies any ambiguities in the original description
4. Offers alternative interpretations if the description was ambiguous
5. Asks for confirmation if critical assumptions were made

Be friendly, clear, and focus on building trust. Use plain English, avoid jargon when possible.
Format your response in a structured way that's easy to read.
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

Please provide a clear explanation covering:
1. **How I interpreted this strategy** - Summarize the key elements (ticker, dates, entry/exit rules)
2. **Assumptions I made** - Any default values, implicit rules, or interpretations I added
3. **Ambiguities I noticed** - Parts of your description that could be interpreted multiple ways
4. **Alternative interpretations** - If there were ambiguities, what other ways could this have been interpreted?
5. **Confirmation needed** - Are there any critical assumptions that need your confirmation before proceeding?

Format your response in a friendly, conversational tone. Use bullet points or sections for clarity. If everything is clear and unambiguous, you can be brief."""

    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content

