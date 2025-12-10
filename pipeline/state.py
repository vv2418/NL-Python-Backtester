from __future__ import annotations

from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any

import pandas as pd

from core.strategy_spec import StrategySpec
from core.validator import ValidationResult
from core.backtester import Trade


class PipelineState(TypedDict):
    """State schema for the agentic pipeline.
    
    This TypedDict defines all state variables that flow through the pipeline,
    from initial user input to final results.
    """
    
    # Input parameters
    user_text: str  # Natural language strategy description
    model: str  # LLM model to use (gpt-4o-mini, gpt-4o, etc.)
    session_id: str  # Unique session identifier
    timestamp: datetime  # Pipeline start timestamp
    
    # Intermediate results - Translation & Interpretation
    spec: Optional[StrategySpec]  # Parsed strategy specification
    interpretation: Optional[str]  # Human-readable interpretation explanation
    confirmed: bool  # Whether user has confirmed the interpretation
    
    # Validation results
    validation_result: Optional[ValidationResult]  # Structure validation result
    data_validation_result: Optional[ValidationResult]  # Data-dependent validation result
    
    # Data processing
    data: Optional[pd.DataFrame]  # Price data with features (OHLCV, MAs, volatility)
    
    # Backtest results
    backtest_results: Optional[pd.DataFrame]  # Backtest output with positions, returns, equity curve
    metrics: Optional[Dict[str, float]]  # Performance metrics (CAGR, drawdown, Sharpe, etc.)
    trades: Optional[List[Trade]]  # Detailed trade-by-trade information
    explanation: Optional[str]  # LLM-generated explanation of results
    
    # Error tracking
    errors: List[str]  # List of error messages encountered
    warnings: List[str]  # List of warning messages
    
    # Metrics tracking
    metrics_log: List[Dict[str, Any]]  # Log of all LLM calls (tokens, costs, latency)
    
    # Pipeline control
    current_step: str  # Current pipeline step name
    retry_count: Dict[str, int]  # Track retry attempts per node


def create_initial_state(
    user_text: str,
    model: str = "gpt-4o-mini",
    session_id: Optional[str] = None
) -> PipelineState:
    """Create initial pipeline state from user input.
    
    Args:
        user_text: Natural language strategy description
        model: LLM model to use
        session_id: Optional session identifier (generated if not provided)
    
    Returns:
        Initialized PipelineState with all required fields set
    """
    import uuid
    
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    return PipelineState(
        # Input
        user_text=user_text,
        model=model,
        session_id=session_id,
        timestamp=datetime.now(),
        
        # Intermediate results
        spec=None,
        interpretation=None,
        confirmed=False,
        
        # Validation
        validation_result=None,
        data_validation_result=None,
        
        # Data
        data=None,
        
        # Results
        backtest_results=None,
        metrics=None,
        trades=None,
        explanation=None,
        
        # Error tracking
        errors=[],
        warnings=[],
        
        # Metrics
        metrics_log=[],
        
        # Control
        current_step="initialize",
        retry_count={},
    )


# Quick validation when run directly: python pipeline/state.py
if __name__ == "__main__":
    print("✓ Testing state.py...")
    state = create_initial_state("Test strategy", "gpt-4o-mini")
    assert state["user_text"] == "Test strategy"
    assert state["model"] == "gpt-4o-mini"
    assert state["spec"] is None
    assert state["confirmed"] is False
    assert len(state["errors"]) == 0
    assert state["current_step"] == "initialize"
    print("✓ State schema validated successfully!")

