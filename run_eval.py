import json
import os
from openai import OpenAI
from llm.translator import translate_to_spec
from core.strategy_spec import StrategySpec

# -----------------------------
# CONFIG
# -----------------------------
MODELS = [
    "gpt-4o-mini",
    "gpt-4o"
]

# Load prompts
from prompts import PROMPTS


# -----------------------------
# GPT SYSTEM PROMPT (schema enforced)
# -----------------------------
SYSTEM_PROMPT = """
You are a translator converting natural-language trading strategies into JSON.

Output MUST follow exactly this schema:

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

RULES:
- The root key is 'ticker' (NOT 'symbol').
- Only allowed entry/exit rule types: "crossover" and "vol_filter".
- Metrics must always be present.
- Dates must be ISO strings.
- Output ONLY JSON — no markdown or explanation.
"""


GROUND_TRUTH_MODEL = "gpt-4o-mini"   

def normalize(obj):
    """Normalize dict/list structure so ordering never affects comparison."""
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in sorted(obj.items())}

    if isinstance(obj, list):

        if all(isinstance(x, dict) for x in obj):
            # Sort list of dicts by stable JSON representation
            return sorted([normalize(x) for x in obj],
                          key=lambda d: json.dumps(d, sort_keys=True))

        if all(not isinstance(x, dict) for x in obj):
            # Sort scalars directly
            return sorted(obj)

    return obj


# -----------------------------
# COMPARISON FUNCTION
# -----------------------------
def compare_json(gt, pred):
    """Order-insensitive comparison of ground-truth vs GPT output."""

    if pred is None:
        return {
            "valid_json": False,
            "correct_symbol": False,
            "correct_dates": False,
            "correct_entry_rules": False,
            "correct_exit_rules": False,
            "correct_metrics": False,
            "full_match": False
        }

    # Accept both ticker/symbol names
    gt_symbol = gt.get("ticker") or gt.get("symbol")
    pred_symbol = pred.get("ticker") or pred.get("symbol")

    # Normalize structures
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)

    score = {
        "valid_json": True,
        "correct_symbol": (gt_symbol == pred_symbol),
        "correct_dates": (
            gt_norm.get("start_date") == pred_norm.get("start_date")
            and gt_norm.get("end_date") == pred_norm.get("end_date")
        ),
        "correct_entry_rules":
            normalize(gt_norm.get("entry_rules")) == normalize(pred_norm.get("entry_rules")),
        "correct_exit_rules":
            normalize(gt_norm.get("exit_rules")) == normalize(pred_norm.get("exit_rules")),
        "correct_metrics":
            normalize(gt_norm.get("metrics")) == normalize(pred_norm.get("metrics")),
    }

    score["full_match"] = all(score.values())
    return score


# -----------------------------
# GPT CALL
# -----------------------------
def run_gpt(model, prompt):
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
    except Exception:
        return None

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        return None


# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------
if __name__ == "__main__":

    results = []

    print("\n=== Running NL → DSL Evaluation ===\n")

    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"\n--- Prompt {i}/{len(PROMPTS)} ---")

        # Ground truth (YOUR translator)
        gt_spec: StrategySpec = translate_to_spec(prompt, model=GROUND_TRUTH_MODEL)

        gt_json = gt_spec.to_dict()
        
        row = {
            "prompt": prompt,
            "ground_truth": gt_json,
            "models": {}
        }

        for model in MODELS:
            print(f"  Evaluating {model}...")

            pred_json = run_gpt(model, prompt)
            score = compare_json(gt_json, pred_json)

            row["models"][model] = {
                "output": pred_json,
                "score": score
            }

        results.append(row)

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved evaluation to eval_results.json")
