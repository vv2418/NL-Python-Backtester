from __future__ import annotations

import logging
import os

import streamlit as st

from core.strategy_spec import StrategySpec
from core.data import load_price_data, add_features
from core.backtester import run_backtest
from core.metrics import compute_basic_metrics
from core.plotting import plot_equity_curve
from llm.translator import translate_to_spec
from llm.explainer import summarize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


EXAMPLE_STRATEGY = """Backtest AAPL from 2018-01-01 to 2024-01-01.
Go long when the 10-day moving average crosses above the 50-day moving average.
Exit when the 10-day moving average crosses back below the 50-day.
Only enter new positions when 20-day realized volatility is below its 1-year median.
Show CAGR, max drawdown, and Sharpe ratio."""


def main():
    st.set_page_config(page_title="Backtest Chat Copilot", layout="wide")
    st.title("Backtest Chat Copilot")

    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    if not api_key_set:
        st.warning(
            "OPENAI_API_KEY is not set in your environment. "
            "Set it before running the app to enable LLM features."
        )

    st.markdown(
        "Describe a single-asset, long-only strategy in plain English. "
        "The app will translate it into a backtest spec, run the backtest, "
        "and summarize the performance."
    )

    user_text = st.text_area(
        "Strategy description",
        value=EXAMPLE_STRATEGY,
        height=220,
    )

    run_button = st.button("Run backtest")

    if not run_button:
        return

    # 1. Translate NL -> StrategySpec
    with st.spinner("Translating strategy with LLM..."):
        try:
            spec: StrategySpec = translate_to_spec(user_text)
            logger.info("Translation completed successfully")
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            st.error(f"Failed to translate strategy: {e}")
            return

    st.subheader("Parsed Strategy Spec")
    st.json(spec.to_dict())

    # 2. Load data, add features, run backtest
    with st.spinner("Loading data and running backtest..."):
        try:
            df = load_price_data(spec)
            logger.info("Data fetching completed successfully")
            df = add_features(df, spec)
            logger.info("Feature addition completed successfully")
            results_df = run_backtest(df, spec)
            logger.info("Backtesting completed successfully")
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            st.error(f"Backtest failed: {e}")
            return

    metrics = compute_basic_metrics(results_df)
    logger.info("Metrics computation completed successfully")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Equity Curve")
        fig = plot_equity_curve(results_df)
        logger.info("Plotting completed successfully")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Performance Metrics")
        st.json(metrics)

        with st.spinner("Generating explanation..."):
            try:
                explanation = summarize_results(spec, metrics)
                logger.info("Explanation generation completed successfully")
            except Exception as e:
                logger.error(f"Explanation generation failed: {e}")
                explanation = f"(Explanation failed: {e})"

        st.subheader("LLM Explanation")
        st.write(explanation)


if __name__ == "__main__":
    main()
