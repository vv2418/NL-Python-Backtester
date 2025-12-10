from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Pricing per 1M tokens (as of 2024, update as needed)
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

LOG_FILE = Path(__file__).parent.parent / "metrics_log.csv"


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for token usage."""
    if model not in MODEL_PRICING:
        return 0.0
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def log_metrics(
    task_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_seconds: float,
    success: bool,
    error_message: Optional[str] = None,
) -> None:
    """Log metrics to CSV file."""
    total_tokens = input_tokens + output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)
    timestamp = datetime.now().isoformat()

    # Create log file if it doesn't exist
    file_exists = LOG_FILE.exists()
    
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow([
                "timestamp",
                "task_type",
                "model",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cost_usd",
                "latency_seconds",
                "success",
                "error_message",
            ])
        
        writer.writerow([
            timestamp,
            task_type,
            model,
            input_tokens,
            output_tokens,
            total_tokens,
            f"{cost:.6f}",
            f"{latency_seconds:.3f}",
            success,
            error_message or "",
        ])

