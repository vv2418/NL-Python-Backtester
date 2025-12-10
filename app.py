from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Dict

import streamlit as st

from core.strategy_spec import StrategySpec, CrossoverRule, VolFilterRule
from core.backtester import trades_to_dataframe
from core.plotting import plot_equity_curve
from pipeline.graph import run_pipeline, resume_pipeline, get_pipeline_state

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

    # Model selection dropdown
    available_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"
    
    selected_model = st.selectbox(
        "🤖 LLM Model",
        available_models,
        index=available_models.index(st.session_state.selected_model),
        help="Select the OpenAI model to use for translation, interpretation, and explanation. gpt-4o-mini is fastest and cheapest, gpt-4o offers better quality.",
    )
    st.session_state.selected_model = selected_model

    user_text = st.text_area(
        "Strategy description",
        value=EXAMPLE_STRATEGY,
        height=220,
    )

    # Initialize pipeline session state
    if "pipeline_session_id" not in st.session_state:
        st.session_state.pipeline_session_id = str(uuid.uuid4())
    if "pipeline_state" not in st.session_state:
        st.session_state.pipeline_state = None
    if "user_text_hash" not in st.session_state:
        st.session_state.user_text_hash = None
    if "model_used" not in st.session_state:
        st.session_state.model_used = None

    # Hash user text to detect changes
    current_text_hash = hashlib.md5(user_text.encode()).hexdigest()
    
    # Reset pipeline if user text or model changed
    model_changed = st.session_state.model_used != selected_model
    if st.session_state.user_text_hash != current_text_hash or model_changed:
        st.session_state.pipeline_state = None
        st.session_state.pipeline_session_id = str(uuid.uuid4())  # New session for new input
        st.session_state.user_text_hash = current_text_hash
        st.session_state.model_used = selected_model

    run_button = st.button("Run backtest")

    # Run pipeline or get existing state
    if run_button or st.session_state.pipeline_state is None:
        if run_button:
            with st.spinner(f"Running pipeline with {selected_model}..."):
                try:
                    # Start pipeline (will pause at checkpoint)
                    state = run_pipeline(
                        user_text=user_text,
                        model=selected_model,
                        session_id=st.session_state.pipeline_session_id,
                        confirmed=False
                    )
                    st.session_state.pipeline_state = state
                    logger.info("Pipeline started, paused at checkpoint")
                except Exception as e:
                    logger.error(f"Pipeline execution failed: {e}")
                    st.error(f"Pipeline failed: {e}")
                    # Try to get state from checkpoint
                    try:
                        state = get_pipeline_state(st.session_state.pipeline_session_id)
                        if state:
                            st.session_state.pipeline_state = state
                    except:
                        pass
                    st.stop()
        else:
            # Try to get existing state from checkpoint
            state = get_pipeline_state(st.session_state.pipeline_session_id)
            if state:
                st.session_state.pipeline_state = state
            else:
                st.session_state.pipeline_state = None

    # Get current state
    state = st.session_state.pipeline_state
    
    if state is None:
        st.info("👆 Enter a strategy description and click 'Run backtest' to start.")
        st.stop()

    # Check if pipeline is waiting for confirmation (at checkpoint)
    if not state.get("confirmed", False) and state.get("interpretation"):
        # Show interpretation and wait for confirmation
        spec = state.get("spec")
        interpretation = state.get("interpretation")
        
        if spec:
            # Get default assumptions
            default_assumptions = get_default_assumptions(spec)
            
            # Display interpretation and assumptions side-by-side
            if interpretation:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 How I Interpreted Your Strategy")
                    with st.expander("View interpretation details", expanded=True):
                        st.markdown(interpretation)
                
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
                st.json(spec.to_dict())

            # Show validation errors/warnings if any
            validation_result = state.get("validation_result")
            if validation_result and (validation_result.errors or validation_result.warnings):
                st.subheader("⚠️ Strategy Check")
                for msg in validation_result.errors:
                    st.error(msg)
                for msg in validation_result.warnings:
                    st.warning(msg)
                if validation_result.errors:
                    st.error("❌ Please fix the errors above. Edit your strategy and run again.")
                    st.stop()

            # Human-in-the-loop: Require confirmation before proceeding
            st.markdown("---")
            st.info("👆 Please review the interpretation and specification above. If everything looks correct, confirm below to proceed with the backtest.")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                confirm_button = st.button("✅ Confirm & Proceed", type="primary", use_container_width=True)
            with col2:
                st.caption("Click to proceed with backtesting, or edit your strategy description above and run again.")
            
            if confirm_button:
                # Resume pipeline after confirmation
                with st.spinner("Resuming pipeline..."):
                    try:
                        final_state = resume_pipeline(
                            session_id=st.session_state.pipeline_session_id,
                            confirmed=True
                        )
                        if final_state:
                            logger.info(f"Pipeline resumed - current_step: {final_state.get('current_step')}, has_backtest_results: {final_state.get('backtest_results') is not None}")
                            st.session_state.pipeline_state = final_state
                            logger.info("Pipeline resumed after confirmation")
                            # Rerun to show updated state
                            st.rerun()
                        else:
                            st.error("Failed to resume pipeline - no state returned")
                            logger.error("resume_pipeline returned None")
                            st.stop()
                    except Exception as e:
                        logger.error(f"Pipeline resume failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        st.error(f"Failed to resume pipeline: {e}")
                        st.stop()
            
            st.stop()  # Stop execution until user confirms
    
    # Pipeline has completed or is in progress after confirmation
    # Check if we have final results
    if state.get("backtest_results") is not None:
        # Pipeline completed - show results
        spec = state.get("spec")
        results_df = state.get("backtest_results")
        metrics = state.get("metrics", {})
        trades = state.get("trades", [])
        explanation = state.get("explanation", "")
        
        # Show validation warnings if any
        validation_result = state.get("validation_result")
        data_validation_result = state.get("data_validation_result")
        
        if validation_result and validation_result.warnings:
            st.subheader("⚠️ Strategy Check")
            for msg in validation_result.warnings:
                st.warning(msg)
        
        if data_validation_result and (data_validation_result.errors or data_validation_result.warnings):
            st.subheader("Pre-backtest QA")
            for msg in data_validation_result.errors:
                st.error(msg)
            for msg in data_validation_result.warnings:
                st.warning(msg)
        
        # Show errors if any
        if state.get("errors"):
            st.subheader("⚠️ Errors")
            for error in state["errors"]:
                st.error(error)
        
        # Show results
        left, right = st.columns([2, 1])

        with left:
            st.subheader("Equity Curve")
            fig = plot_equity_curve(results_df)
            logger.info("Plotting completed successfully")
            st.pyplot(fig, clear_figure=True)

        with right:
            st.subheader("Performance Metrics")
            st.json(metrics)

            if explanation:
                st.subheader("LLM Explanation")
                st.write(explanation)

        # Trade-by-Trade Debugger Section
        st.markdown("---")
        st.subheader("🔍 Trade-by-Trade Debugger")

        if trades:
            trades_df = trades_to_dataframe(trades)
            
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
    else:
        # Pipeline is running after confirmation or completed
        # First, try to refresh state from checkpoint to get latest
        try:
            updated_state = get_pipeline_state(st.session_state.pipeline_session_id)
            if updated_state:
                st.session_state.pipeline_state = updated_state
                state = updated_state
        except Exception as e:
            logger.warning(f"Could not refresh pipeline state: {e}")
            # Continue with existing state
        
        current_step = state.get("current_step", "unknown")
        
        # Check if pipeline completed by looking for backtest_results
        if state.get("backtest_results") is None:
            # Pipeline still running or hasn't started yet
            st.info(f"⏳ Pipeline running... Current step: {current_step}")
            
            # Check for errors
            if state.get("errors"):
                st.error("Pipeline execution encountered errors:")
                for error in state["errors"]:
                    st.error(error)
                st.stop()
            
            # Show a button to manually refresh
            if st.button("🔄 Refresh Status"):
                # Force refresh from checkpoint
                try:
                    logger.info(f"Refreshing pipeline state for session {st.session_state.pipeline_session_id}")
                    refreshed_state = get_pipeline_state(st.session_state.pipeline_session_id)
                    if refreshed_state:
                        logger.info(f"Refreshed state - current_step: {refreshed_state.get('current_step')}, has_backtest_results: {refreshed_state.get('backtest_results') is not None}")
                        st.session_state.pipeline_state = refreshed_state
                        st.rerun()
                    else:
                        st.warning("No pipeline state found. Please run the pipeline again.")
                        logger.warning("No state found when refreshing")
                except Exception as e:
                    logger.error(f"Error refreshing pipeline state: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    st.error(f"Error refreshing state: {e}")
            
            st.stop()  # Stop here to prevent infinite reruns
        else:
            # Pipeline completed - results should be shown above
            # This shouldn't be reached if backtest_results exists
            st.rerun()


if __name__ == "__main__":
    main()
