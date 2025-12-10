from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes import (
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
)
from pipeline.checkpoints import create_checkpointer, check_confirmation

logger = logging.getLogger(__name__)


def check_validation(state: PipelineState) -> Literal["pass", "fail"]:
    """Conditional routing: check if validation passed."""
    if state.get("validation_result") is None:
        return "fail"
    
    validation_result = state["validation_result"]
    if validation_result.ok and len(validation_result.errors) == 0:
        return "pass"
    return "fail"


def check_data_validation(state: PipelineState) -> Literal["pass", "fail"]:
    """Conditional routing: check if data validation passed."""
    if state.get("data_validation_result") is None:
        return "fail"
    
    validation_result = state["data_validation_result"]
    if validation_result.ok and len(validation_result.errors) == 0:
        return "pass"
    return "fail"


def checkpoint_reducer(state: PipelineState) -> PipelineState:
    """Reducer to prepare state for checkpointing.
    
    Converts non-serializable objects (DataFrames) to None before checkpointing.
    DataFrames will be regenerated when needed, or we can store them separately.
    """
    # Create a copy of state without DataFrames for checkpointing
    checkpoint_state = dict(state)
    
    # Convert DataFrames to None for checkpointing (they're not serializable)
    if isinstance(checkpoint_state.get("data"), pd.DataFrame):
        checkpoint_state["data"] = None
    if isinstance(checkpoint_state.get("backtest_results"), pd.DataFrame):
        checkpoint_state["backtest_results"] = None
    
    return checkpoint_state


def create_pipeline():
    """Create and compile the LangGraph pipeline workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    workflow = StateGraph(PipelineState)
    
    # Add all nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("interpret", interpret_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("add_features", add_features_node)
    workflow.add_node("pre_qa", pre_qa_node)
    workflow.add_node("backtest", backtest_node)
    workflow.add_node("metrics", metrics_node)
    workflow.add_node("trades", trades_node)
    workflow.add_node("explain", explain_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("persist", persist_node)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add sequential edges
    workflow.add_edge("initialize", "translate")
    workflow.add_edge("translate", "interpret")
    workflow.add_edge("interpret", "validate")
    
    # Note: interrupt_before=["validate"] will pause execution here
    # When resuming, execution continues to validate node
    
    # Conditional edge: validation check
    workflow.add_conditional_edges(
        "validate",
        check_validation,
        {
            "pass": "fetch_data",
            "fail": END,
        }
    )
    
    # Continue sequential flow after validation
    workflow.add_edge("fetch_data", "add_features")
    workflow.add_edge("add_features", "pre_qa")
    
    # Conditional edge: data validation check
    workflow.add_conditional_edges(
        "pre_qa",
        check_data_validation,
        {
            "pass": "backtest",
            "fail": END,
        }
    )
    
    # Continue sequential flow after data validation
    workflow.add_edge("backtest", "metrics")
    
    # Metrics and trades can run in parallel (but we'll do sequential for simplicity)
    workflow.add_edge("metrics", "trades")
    workflow.add_edge("trades", "explain")
    workflow.add_edge("explain", "aggregate")
    workflow.add_edge("aggregate", "persist")
    workflow.add_edge("persist", END)
    
    # Compile the graph with checkpointer for human-in-the-loop
    # Interrupt before validate node to wait for user confirmation
    checkpointer = create_checkpointer()
    
    # Note: We can't use a reducer directly in compile, but we'll handle
    # DataFrame serialization by catching the error and continuing
    # The state will still be returned, just not fully checkpointed after DataFrames are added
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["validate"]  # Pause here for user confirmation
    )
    
    logger.info("Pipeline graph created and compiled successfully with checkpointing")
    return graph


def run_pipeline(
    user_text: str,
    model: str = "gpt-4o-mini",
    session_id: str | None = None,
    confirmed: bool = False
):
    """Run the complete pipeline with given inputs.
    
    Args:
        user_text: Natural language strategy description
        model: LLM model to use
        session_id: Optional session identifier (used as thread_id for checkpointing)
        confirmed: Whether user has confirmed interpretation (for resuming from checkpoint)
    
    Returns:
        Final pipeline state with all results, or state at checkpoint if waiting for confirmation
    """
    from pipeline.state import create_initial_state
    from pipeline.checkpoints import resume_from_checkpoint
    
    # Create initial state
    initial_state = create_initial_state(user_text, model, session_id)
    initial_state["confirmed"] = confirmed
    
    # Use session_id as thread_id for checkpointing
    thread_id = session_id or initial_state["session_id"]
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create graph
    graph = create_pipeline()
    
    # Run graph (will interrupt before validate if not confirmed)
    try:
        final_state = graph.invoke(initial_state, config)
        
        # Check if execution was interrupted (waiting for confirmation)
        current_state = graph.get_state(config)
        if current_state and current_state.next:
            # Execution was interrupted - waiting at checkpoint
            logger.info("Pipeline paused at checkpoint, waiting for user confirmation")
            return current_state.values
        
        return final_state
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        # Return current state if available
        current_state = graph.get_state(config)
        if current_state:
            return current_state.values
        raise


def get_pipeline_state(session_id: str):
    """Get the current state of a pipeline execution.
    
    Args:
        session_id: Session/thread identifier
    
    Returns:
        Current state from checkpoint, or None if not found
    """
    from pipeline.checkpoints import get_checkpoint_state, create_checkpointer
    
    # Create a graph instance to access the shared checkpointer
    # The checkpointer is shared, so we can get state from any graph instance
    graph = create_pipeline()
    try:
        return get_checkpoint_state(graph, session_id)
    except Exception as e:
        logger.error(f"Error getting pipeline state for session {session_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def resume_pipeline(session_id: str, confirmed: bool = True):
    """Resume pipeline execution from checkpoint after user confirmation.
    
    Args:
        session_id: Session/thread identifier
        confirmed: Whether user confirmed (True) or wants to reset (False)
    
    Returns:
        Updated state after resuming, or current state if still waiting
    """
    graph = create_pipeline()
    config = {"configurable": {"thread_id": session_id}}
    
    # Get current state
    try:
        current_state = graph.get_state(config)
        if not current_state:
            logger.error(f"No checkpoint found for session {session_id}")
            raise ValueError(f"No checkpoint found for session {session_id}")
    except Exception as e:
        logger.error(f"Error getting checkpoint state: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info(f"Found checkpoint state for session {session_id}, current step: {current_state.values.get('current_step')}")
    
    # Update confirmed status
    current_state.values["confirmed"] = confirmed
    try:
        graph.update_state(config, current_state.values)
        logger.info("Updated confirmed status in checkpoint")
    except Exception as e:
        logger.warning(f"Could not update state: {e}, continuing anyway")
    
    # Resume execution
    if confirmed:
        try:
            # Invoke with None to continue from checkpoint
            # This will run the pipeline from the checkpoint to completion
            logger.info(f"Resuming pipeline execution for session {session_id}")
            
            # Use stream to handle checkpoint errors gracefully
            # The pipeline will complete but checkpointing may fail on DataFrames
            final_state = None
            try:
                # Use stream() instead of invoke() to capture state even if checkpointing fails
                final_state = None
                for event in graph.stream(None, config):
                    if event:
                        for node_name, node_state in event.items():
                            if node_state and isinstance(node_state, dict):
                                final_state = node_state
                                logger.debug(f"Captured state from node: {node_name}")
            except Exception as checkpoint_error:
                # Checkpoint error (likely DataFrame serialization) - but pipeline may have completed
                if "msgpack" in str(checkpoint_error).lower() or "serializable" in str(checkpoint_error).lower():
                    logger.warning(f"Checkpoint serialization error (expected for DataFrames): {checkpoint_error}")
                    # final_state should already be captured from stream before error
                else:
                    # Real error - re-raise
                    raise
            
            logger.info(f"Pipeline stream completed. Final state captured: {final_state is not None}")
            
            # Use the final_state from stream (it has DataFrames even if checkpointing failed)
            if final_state and isinstance(final_state, dict):
                has_backtest = final_state.get("backtest_results") is not None
                current_step = final_state.get("current_step", "unknown")
                logger.info(f"Using final_state from stream - current_step: {current_step}, has_backtest_results: {has_backtest}")
                return final_state
            
            # Fallback: try to get state from checkpoint (won't have DataFrames)
            logger.warning("No final_state from stream, trying checkpoint...")
            updated_state = graph.get_state(config)
            
            if updated_state:
                logger.info(f"Using checkpoint state - has backtest_results: {updated_state.values.get('backtest_results') is not None}")
                return updated_state.values
            
            # Last resort
            logger.error("No state found after pipeline execution")
            return current_state.values
            
        except Exception as e:
            logger.error(f"Error resuming pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return current state if available
            try:
                updated_state = graph.get_state(config)
                if updated_state:
                    logger.info("Returning state after error")
                    return updated_state.values
            except Exception as e2:
                logger.error(f"Error getting state after resume failure: {e2}")
            return current_state.values if current_state else None
    else:
        # User wants to reset - return current state without proceeding
        return current_state.values


# Quick validation when run directly: python pipeline/graph.py
if __name__ == "__main__":
    print("✓ Testing graph.py...")
    
    try:
        graph = create_pipeline()
        print("✓ Graph created successfully")
        
        # Test that graph has the expected structure
        nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
        print(f"✓ Graph compiled with {len(nodes)} nodes")
        
        # Test with minimal state
        from pipeline.state import create_initial_state
        
        initial_state = create_initial_state("Test strategy", "gpt-4o-mini")
        print("✓ Initial state created")
        print(f"  - Session ID: {initial_state['session_id']}")
        print(f"  - Model: {initial_state['model']}")
        print(f"  - Current step: {initial_state['current_step']}")
        
        print("✓ Graph structure validated successfully!")
        print("\nNote: Full execution requires OpenAI API key and will make actual LLM calls.")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure langgraph is installed: pip install langgraph")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

