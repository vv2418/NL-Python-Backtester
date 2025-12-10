from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver

from pipeline.state import PipelineState

logger = logging.getLogger(__name__)

# Shared checkpointer instance - singleton pattern
_shared_checkpointer: MemorySaver | None = None


def create_checkpointer():
    """Create or return the shared checkpointer for state persistence.
    
    Uses a singleton pattern to ensure all graph instances share the same
    checkpointer, so state persists across different graph instances.
    
    Returns:
        Shared MemorySaver instance for checkpointing
    """
    global _shared_checkpointer
    if _shared_checkpointer is None:
        _shared_checkpointer = MemorySaver()
        logger.info("Created shared checkpointer instance")
    return _shared_checkpointer


def check_confirmation(state: PipelineState) -> Literal["proceed", "wait"]:
    """Conditional routing: check if user has confirmed interpretation.
    
    Args:
        state: Current pipeline state
    
    Returns:
        "proceed" if confirmed, "wait" if not confirmed
    """
    if state.get("confirmed", False):
        logger.info("User confirmed interpretation, proceeding to validation")
        return "proceed"
    else:
        logger.info("Waiting for user confirmation...")
        return "wait"


def get_checkpoint_state(graph, thread_id: str):
    """Get the current state from a checkpoint.
    
    Args:
        graph: Compiled LangGraph instance
        thread_id: Thread/session identifier
    
    Returns:
        Current state from checkpoint, or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        return state.values if state else None
    except Exception as e:
        logger.error(f"Error getting checkpoint state: {e}")
        return None


def resume_from_checkpoint(graph, thread_id: str, confirmed: bool = True):
    """Resume pipeline execution from checkpoint after user confirmation.
    
    Args:
        graph: Compiled LangGraph instance
        thread_id: Thread/session identifier
        confirmed: Whether user confirmed (True) or wants to reset (False)
    
    Returns:
        Updated state after resuming
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    if confirmed:
        # Update state to mark as confirmed
        current_state = graph.get_state(config)
        if current_state:
            current_state.values["confirmed"] = True
            graph.update_state(config, current_state.values)
        
        # Resume execution
        result = graph.invoke(None, config)
        return result
    else:
        # User wants to reset - would need to restart from translate
        # For now, just return current state
        current_state = graph.get_state(config)
        return current_state.values if current_state else None


# Quick validation when run directly: python pipeline/checkpoints.py
if __name__ == "__main__":
    print("✓ Testing checkpoints.py...")
    
    try:
        checkpointer = create_checkpointer()
        print("✓ Checkpointer created successfully")
        
        # Test confirmation check function
        from pipeline.state import create_initial_state
        
        state_unconfirmed = create_initial_state("Test", "gpt-4o-mini")
        assert check_confirmation(state_unconfirmed) == "wait"
        
        state_confirmed = create_initial_state("Test", "gpt-4o-mini")
        state_confirmed["confirmed"] = True
        assert check_confirmation(state_confirmed) == "proceed"
        
        print("✓ Checkpoint functions validated successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

