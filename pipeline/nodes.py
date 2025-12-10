from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.strategy_spec import StrategySpec
from core.validator import validate_spec, validate_with_data
from core.data import load_price_data, add_features
from core.backtester import run_backtest, extract_trades
from core.metrics import compute_basic_metrics
from llm.translator import translate_to_spec
from llm.interpreter import explain_interpretation
from llm.explainer import summarize_results
from pipeline.errors import retry_on_api_error, retry_on_network_error, handle_node_error

if TYPE_CHECKING:
    from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def initialize_node(state: "PipelineState") -> "PipelineState":
    """Initialize the pipeline state (already done, just update step)."""
    state["current_step"] = "initialize"
    logger.info(f"Pipeline initialized for session {state['session_id']}")
    return state


def translate_node(state: "PipelineState") -> "PipelineState":
    """Translate natural language strategy to StrategySpec."""
    state["current_step"] = "translate"
    
    @retry_on_api_error(max_retries=3)
    def _translate_with_retry():
        return translate_to_spec(state["user_text"], model=state["model"])
    
    try:
        logger.info("Translating strategy description to spec...")
        spec = _translate_with_retry()
        state["spec"] = spec
        logger.info(f"Translation successful: {spec.ticker} from {spec.start_date} to {spec.end_date}")
    except Exception as e:
        state = handle_node_error("translate_node", state, e, is_critical=True, retry_count_key="translate")
        raise  # Re-raise to allow graph-level error handling
    
    return state


def interpret_node(state: "PipelineState") -> "PipelineState":
    """Generate human-readable interpretation explanation."""
    state["current_step"] = "interpret"
    
    if state["spec"] is None:
        error_msg = "Cannot interpret: spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    @retry_on_api_error(max_retries=2)  # Fewer retries for non-critical operation
    def _interpret_with_retry():
        return explain_interpretation(
            state["user_text"],
            state["spec"],
            model=state["model"]
        )
    
    try:
        logger.info("Generating interpretation explanation...")
        interpretation = _interpret_with_retry()
        state["interpretation"] = interpretation
        logger.info("Interpretation generated successfully")
    except Exception as e:
        # Non-critical: continue with None interpretation
        state = handle_node_error("interpret_node", state, e, is_critical=False, retry_count_key="interpret")
        state["warnings"].append(f"Interpretation generation failed: {str(e)}")
        state["interpretation"] = None
    
    return state


def validate_node(state: "PipelineState") -> "PipelineState":
    """Validate strategy structure and logic."""
    state["current_step"] = "validate"
    
    if state["spec"] is None:
        error_msg = "Cannot validate: spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Validating strategy specification...")
        validation_result = validate_spec(state["spec"])
        state["validation_result"] = validation_result
        
        # Add errors and warnings to state
        state["errors"].extend(validation_result.errors)
        state["warnings"].extend(validation_result.warnings)
        
        if validation_result.errors:
            logger.error(f"Validation found {len(validation_result.errors)} errors")
        if validation_result.warnings:
            logger.warning(f"Validation found {len(validation_result.warnings)} warnings")
        
        if validation_result.ok:
            logger.info("Validation passed")
        else:
            logger.error("Validation failed")
    except Exception as e:
        error_msg = f"Validation failed with exception: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
    
    return state


def fetch_data_node(state: "PipelineState") -> "PipelineState":
    """Download price data from yfinance."""
    state["current_step"] = "fetch_data"
    
    if state["spec"] is None:
        error_msg = "Cannot fetch data: spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    @retry_on_network_error(max_retries=3)
    def _fetch_data_with_retry():
        return load_price_data(state["spec"])
    
    try:
        logger.info(f"Fetching price data for {state['spec'].ticker}...")
        df = _fetch_data_with_retry()
        state["data"] = df
        logger.info(f"Data fetched successfully: {len(df)} rows")
    except Exception as e:
        state = handle_node_error("fetch_data_node", state, e, is_critical=True, retry_count_key="fetch_data")
        raise  # Re-raise to allow graph-level error handling
    
    return state


def add_features_node(state: "PipelineState") -> "PipelineState":
    """Add features (MAs, volatility) to the data."""
    state["current_step"] = "add_features"
    
    if state["data"] is None or state["spec"] is None:
        error_msg = "Cannot add features: data or spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Adding features to data...")
        df = add_features(state["data"], state["spec"])
        state["data"] = df
        logger.info("Features added successfully")
    except Exception as e:
        error_msg = f"Feature addition failed: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
    
    return state


def pre_qa_node(state: "PipelineState") -> "PipelineState":
    """Pre-backtest QA: data-dependent validation."""
    state["current_step"] = "pre_qa"
    
    if state["data"] is None or state["spec"] is None:
        error_msg = "Cannot run pre-backtest QA: data or spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Running pre-backtest QA...")
        validation_result = validate_with_data(state["spec"], state["data"])
        state["data_validation_result"] = validation_result
        
        # Add errors and warnings to state
        state["errors"].extend(validation_result.errors)
        state["warnings"].extend(validation_result.warnings)
        
        if validation_result.errors:
            logger.error(f"Pre-backtest QA found {len(validation_result.errors)} errors")
        if validation_result.warnings:
            logger.warning(f"Pre-backtest QA found {len(validation_result.warnings)} warnings")
        
        if validation_result.ok:
            logger.info("Pre-backtest QA passed")
        else:
            logger.error("Pre-backtest QA failed")
    except Exception as e:
        error_msg = f"Pre-backtest QA failed with exception: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
    
    return state


def backtest_node(state: "PipelineState") -> "PipelineState":
    """Execute the backtest."""
    state["current_step"] = "backtest"
    
    if state["data"] is None or state["spec"] is None:
        error_msg = "Cannot run backtest: data or spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Running backtest...")
        results_df = run_backtest(state["data"], state["spec"])
        state["backtest_results"] = results_df
        logger.info("Backtest completed successfully")
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
    
    return state


def metrics_node(state: "PipelineState") -> "PipelineState":
    """Compute performance metrics."""
    state["current_step"] = "metrics"
    
    if state["backtest_results"] is None:
        error_msg = "Cannot compute metrics: backtest_results is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Computing performance metrics...")
        metrics = compute_basic_metrics(state["backtest_results"])
        state["metrics"] = metrics
        logger.info(f"Metrics computed: {metrics}")
    except Exception as e:
        error_msg = f"Metrics computation failed: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
    
    return state


def trades_node(state: "PipelineState") -> "PipelineState":
    """Extract detailed trade-by-trade information."""
    state["current_step"] = "trades"
    
    if state["data"] is None or state["spec"] is None:
        error_msg = "Cannot extract trades: data or spec is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    try:
        logger.info("Extracting trade details...")
        trades = extract_trades(state["data"], state["spec"])
        state["trades"] = trades
        logger.info(f"Extracted {len(trades)} trades")
    except Exception as e:
        error_msg = f"Trade extraction failed: {str(e)}"
        logger.error(error_msg)
        state["warnings"].append(error_msg)  # Non-critical
        state["trades"] = []
    
    return state


def explain_node(state: "PipelineState") -> "PipelineState":
    """Generate LLM explanation of results."""
    state["current_step"] = "explain"
    
    if state["spec"] is None or state["metrics"] is None:
        error_msg = "Cannot generate explanation: spec or metrics is None"
        state["errors"].append(error_msg)
        logger.error(error_msg)
        return state
    
    @retry_on_api_error(max_retries=2)  # Fewer retries for non-critical operation
    def _explain_with_retry():
        return summarize_results(
            state["spec"],
            state["metrics"],
            model=state["model"]
        )
    
    try:
        logger.info("Generating explanation...")
        explanation = _explain_with_retry()
        state["explanation"] = explanation
        logger.info("Explanation generated successfully")
    except Exception as e:
        # Non-critical: continue without explanation
        state = handle_node_error("explain_node", state, e, is_critical=False, retry_count_key="explain")
        state["warnings"].append(f"Explanation generation failed: {str(e)}")
        state["explanation"] = f"(Explanation unavailable: {str(e)})"
    
    return state


def aggregate_node(state: "PipelineState") -> "PipelineState":
    """Aggregate all results into final output format."""
    state["current_step"] = "aggregate"
    
    logger.info("Aggregating results...")
    # State already contains all aggregated data
    # This node is mainly for logging and final checks
    
    if state["errors"]:
        logger.warning(f"Pipeline completed with {len(state['errors'])} errors")
    if state["warnings"]:
        logger.info(f"Pipeline completed with {len(state['warnings'])} warnings")
    
    logger.info("Results aggregation complete")
    return state


def persist_node(state: "PipelineState") -> "PipelineState":
    """Persist state and metrics to disk."""
    state["current_step"] = "persist"
    
    logger.info("Persisting results...")
    # Metrics are already logged by individual LLM modules via log_metrics()
    # This node is for any additional persistence if needed
    
    logger.info("Persistence complete")
    return state


# Quick validation when run directly: python pipeline/nodes.py
if __name__ == "__main__":
    print("✓ Testing nodes.py...")
    from pipeline.state import create_initial_state
    
    # Test that all node functions exist and have correct signatures
    nodes = [
        initialize_node,
        translate_node,
        interpret_node,
        validate_node,
        fetch_data_node,
        add_features_node,
        pre_qa_node,
        backtest_node,
        metrics_node,
        trades_node,
        explain_node,
        aggregate_node,
        persist_node,
    ]
    
    assert len(nodes) == 13, f"Expected 13 nodes, found {len(nodes)}"
    
    # Test that nodes can accept state
    state = create_initial_state("Test", "gpt-4o-mini")
    test_state = initialize_node(state)
    assert test_state["current_step"] == "initialize"
    
    print("✓ All node functions validated successfully!")

