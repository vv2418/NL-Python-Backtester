"""Basic analysis script for LLM metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

LOG_FILE = Path(__file__).parent.parent / "metrics_log.csv"
OUTPUT_FILE = Path(__file__).parent.parent / "analysis_summary.csv"


def analyze_metrics() -> None:
    """Analyze metrics from log file and generate summary."""
    if not LOG_FILE.exists():
        print(f"No log file found at {LOG_FILE}")
        print("Run some backtests first to generate metrics.")
        return

    df = pd.read_csv(LOG_FILE)

    # Basic statistics by task type and model
    summary = []
    
    for task_type in df["task_type"].unique():
        for model in df["model"].unique():
            subset = df[(df["task_type"] == task_type) & (df["model"] == model)]
            
            if len(subset) == 0:
                continue
            
            summary.append({
                "task_type": task_type,
                "model": model,
                "count": len(subset),
                "success_rate": subset["success"].mean() * 100,
                "avg_input_tokens": subset["input_tokens"].mean(),
                "avg_output_tokens": subset["output_tokens"].mean(),
                "avg_total_tokens": subset["total_tokens"].mean(),
                "total_cost_usd": subset["cost_usd"].astype(float).sum(),
                "avg_cost_usd": subset["cost_usd"].astype(float).mean(),
                "avg_latency_seconds": subset["latency_seconds"].astype(float).mean(),
                "total_latency_seconds": subset["latency_seconds"].astype(float).sum(),
            })

    summary_df = pd.DataFrame(summary)
    
    # Sort by task_type and model
    summary_df = summary_df.sort_values(["task_type", "model"])
    
    # Save summary
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Analysis complete! Summary saved to {OUTPUT_FILE}")
    print("\nSummary Statistics:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Overall totals
    print("\n" + "=" * 80)
    print("Overall Totals:")
    print(f"Total runs: {len(df)}")
    print(f"Total cost: ${df['cost_usd'].astype(float).sum():.4f}")
    print(f"Total tokens: {df['total_tokens'].sum():,}")
    print(f"Overall success rate: {df['success'].mean() * 100:.1f}%")


if __name__ == "__main__":
    analyze_metrics()

