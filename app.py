from __future__ import annotations

import logging
import os
from typing import Dict

import streamlit as st

from core.strategy_spec import StrategySpec, CrossoverRule, VolFilterRule
from core.data import load_price_data, add_features
from core.backtester import run_backtest, extract_trades, trades_to_dataframe
from core.metrics import compute_basic_metrics
from core.plotting import plot_equity_curve
from core.validator import validate_spec, validate_with_data
from llm.translator import translate_to_spec
from llm.explainer import summarize_results
from llm.interpreter import explain_interpretation

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


def get_default_assumptions(spec: StrategySpec) -> Dict[str, str]:
    """Extract and format default assumptions from the strategy spec."""
    assumptions = {}
    
    # Moving average assumptions
    ma_windows = set()
    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, CrossoverRule):
            ma_windows.add(rule.fast_ma)
            ma_windows.add(rule.slow_ma)
    
    if ma_windows:
        ma_list = ", ".join([f"{w}-day" for w in sorted(ma_windows)])
        assumptions["Moving Average Type"] = "Simple Moving Average (SMA)"
        assumptions["MA Windows"] = ma_list
    
    # Volatility assumptions
    vol_windows = set()
    for rule in spec.entry_rules + spec.exit_rules:
        if isinstance(rule, VolFilterRule):
            vol_windows.add(rule.window)
    
    if vol_windows:
        vol_list = ", ".join([f"{w}-day" for w in sorted(vol_windows)])
        assumptions["Realized Volatility Calculation"] = (
            f"Daily returns, {vol_list} rolling window, annualized (√252 scaling)"
        )
        assumptions["1-Year Median Calculation"] = (
            "Rolling 252 trading-day median (trailing window, not calendar year)"
        )
    
    # Execution assumptions
    assumptions["Order Execution"] = "Close-to-close (position changes at market close)"
    assumptions["Position Sizing"] = "Long-only, full position (1.0 when in market, 0.0 when out)"
    assumptions["Entry Logic"] = "All entry rules must be satisfied (AND logic)"
    assumptions["Exit Logic"] = "Any exit rule triggers exit (OR logic)"
    
    # Data assumptions
    assumptions["Price Data"] = "Adjusted close prices from Yahoo Finance"
    assumptions["Return Calculation"] = "Close-to-close percentage returns"
    
    return assumptions


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

    # Initialize session state
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "spec" not in st.session_state:
        st.session_state.spec = None
    if "interpretation" not in st.session_state:
        st.session_state.interpretation = None
    if "user_text_hash" not in st.session_state:
        st.session_state.user_text_hash = None

    # Hash user text to detect changes
    import hashlib
    current_text_hash = hashlib.md5(user_text.encode()).hexdigest()
    
    # Reset confirmation if user text changed
    if st.session_state.user_text_hash != current_text_hash:
        st.session_state.confirmed = False
        st.session_state.spec = None
        st.session_state.interpretation = None
        st.session_state.user_text_hash = current_text_hash

    run_button = st.button("Run backtest")

    # Safety: if confirmed but no spec, reset confirmation
    if st.session_state.confirmed and st.session_state.spec is None:
        st.session_state.confirmed = False
    
    # If user hasn't confirmed yet, show interpretation and wait for confirmation
    if not st.session_state.confirmed:
        if run_button:
            # Translate strategy
            with st.spinner("Translating strategy with LLM..."):
                try:
                    st.session_state.spec = translate_to_spec(user_text)
                    logger.info("Translation completed successfully")
                except Exception as e:
                    logger.error(f"Translation failed: {e}")
                    st.error(f"Failed to translate strategy: {e}")
                    return

            # Generate interpretation explanation
            with st.spinner("Generating interpretation explanation..."):
                try:
                    st.session_state.interpretation = explain_interpretation(
                        user_text, st.session_state.spec
                    )
                    logger.info("Interpretation explanation generated successfully")
                except Exception as e:
                    logger.error(f"Interpretation explanation failed: {e}")
                    st.warning(f"Could not generate interpretation explanation: {e}")
                    st.session_state.interpretation = None

        # Show interpretation if available
        if st.session_state.spec is not None:
            # Get default assumptions
            default_assumptions = get_default_assumptions(st.session_state.spec)
            
            # Display interpretation and assumptions side-by-side
            if st.session_state.interpretation:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 How I Interpreted Your Strategy")
                    with st.expander("View interpretation details", expanded=True):
                        st.markdown(st.session_state.interpretation)
                
                with col2:
                    st.subheader("⚙️ Default Assumptions & Values")
                    with st.expander("View technical defaults", expanded=True):
                        for key, value in default_assumptions.items():
                            st.markdown(f"**{key}:**")
                            st.caption(value)
                            st.markdown("")  # Add spacing

            # Show parsed spec
            st.subheader("📊 Parsed Strategy Specification")
            with st.expander("View technical specification (JSON)", expanded=False):
                st.json(st.session_state.spec.to_dict())

            # Human-in-the-loop: Require confirmation before proceeding
            st.markdown("---")
            st.info("👆 Please review the interpretation and specification above. If everything looks correct, confirm below to proceed with the backtest.")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                confirm_button = st.button("✅ Confirm & Proceed", type="primary", use_container_width=True)
            with col2:
                st.caption("Click to proceed with backtesting, or edit your strategy description above and run again.")
            
            if confirm_button:
                st.session_state.confirmed = True
                st.rerun()  # Rerun to proceed with backtest
            
            st.stop()  # Stop execution until user confirms
    
    # User has confirmed, proceed with backtest
    spec = st.session_state.spec
    
    # Safety check: ensure spec exists
    if spec is None:
        st.error("❌ No strategy specification found. Please click 'Run backtest' first.")
        st.stop()
    
    logger.info("User confirmed interpretation, proceeding with backtest")

    # Run validation after confirmation
    validation = validate_spec(spec)
    if validation.errors or validation.warnings:
        st.subheader("⚠️ Strategy Check")
        for msg in validation.errors:
            st.error(msg)
        for msg in validation.warnings:
            st.warning(msg)
    if validation.errors:
        st.error("❌ Please fix the errors above before proceeding.")
        st.stop()

    with st.spinner("Loading data and running backtest..."):
        try:
            df = load_price_data(spec)
            logger.info("Data fetching completed successfully")
            df = add_features(df, spec)
            logger.info("Feature addition completed successfully")
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            st.error(f"Data preparation failed: {e}")
            return

        data_validation = validate_with_data(spec, df)
        if data_validation.errors or data_validation.warnings:
            st.subheader("Pre-backtest QA")
            for msg in data_validation.errors:
                st.error(msg)
            for msg in data_validation.warnings:
                st.warning(msg)
        if data_validation.errors:
            return

        try:
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

    # Trade-by-Trade Debugger Section
    st.markdown("---")
    st.subheader("🔍 Trade-by-Trade Debugger")

    with st.spinner("Extracting trade details..."):
        try:
            trades = extract_trades(df, spec)
            trades_df = trades_to_dataframe(trades)
            logger.info(f"Extracted {len(trades)} trades")
        except Exception as e:
            logger.error(f"Trade extraction failed: {e}")
            st.error(f"Failed to extract trades: {e}")
            trades = []
            trades_df = None

    if trades_df is not None and len(trades) > 0:
        # Summary stats
        winning_trades = sum(1 for t in trades if t.pnl_pct and t.pnl_pct > 0)
        losing_trades = sum(1 for t in trades if t.pnl_pct and t.pnl_pct < 0)
        avg_pnl = sum(t.pnl_pct for t in trades if t.pnl_pct) / len(trades) if trades else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", len(trades))
        col2.metric("Winning", winning_trades)
        col3.metric("Losing", losing_trades)
        col4.metric("Avg P&L", f"{avg_pnl:+.2f}%")

        # Trade table with expandable details
        with st.expander("View all trades", expanded=True):
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
            )

        # Individual trade explorer
        st.markdown("#### Explore Individual Trade")
        trade_num = st.selectbox(
            "Select trade to inspect",
            options=range(1, len(trades) + 1),
            format_func=lambda x: f"Trade {x}: {trades[x-1].entry_date} → {trades[x-1].exit_date or 'Open'}",
        )

        if trade_num:
            t = trades[trade_num - 1]
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Entry Details**")
                st.write(f"Date: {t.entry_date}")
                st.write(f"Price: ${t.entry_price:.2f}")
                st.info(t.entry_reason)

            with col2:
                st.markdown("**Exit Details**")
                st.write(f"Date: {t.exit_date or 'N/A'}")
                st.write(f"Price: ${t.exit_price:.2f}" if t.exit_price else "N/A")
                if t.exit_reason:
                    st.info(t.exit_reason)
                if t.pnl_pct is not None:
                    if t.pnl_pct >= 0:
                        st.success(f"P&L: {t.pnl_pct:+.2f}%")
                    else:
                        st.error(f"P&L: {t.pnl_pct:+.2f}%")
    else:
        st.info("No trades were executed during this backtest period.")


if __name__ == "__main__":
    main()
