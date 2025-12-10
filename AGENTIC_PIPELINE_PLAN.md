# Agentic Pipeline Plan for Backtest Chat Copilot

## Executive Summary

This document outlines the recommended approach for converting the Backtest Chat Copilot into an agentic pipeline. The system currently uses a Streamlit-based sequential workflow with human-in-the-loop confirmation. An agentic pipeline will provide better state management, error recovery, retry logic, and scalability.

## Recommended Tool: **LangGraph**

### Why LangGraph?

LangGraph is the optimal choice for this project because:

1. **Native State Management**: LangGraph maintains state across multiple LLM calls and processing steps, which is critical for this multi-stage pipeline
2. **Human-in-the-Loop Support**: Built-in support for interrupting workflows for human confirmation, perfect for the interpretation confirmation step
3. **Conditional Logic**: Excellent support for validation gates and conditional routing based on validation results
4. **Error Handling & Retries**: Built-in mechanisms for handling failures and retrying operations
5. **Observability**: Native integration with LangSmith for monitoring and debugging
6. **Python-Native**: Seamless integration with existing Python codebase
7. **Flexible Architecture**: Can work alongside existing Streamlit UI or replace it entirely
8. **Cost Tracking**: Can integrate with existing metrics tracking system

### Alternative Considerations

- **LangChain**: Good for simple chains but lacks sophisticated state management and human-in-the-loop features
- **CrewAI**: Overkill for single-agent workflows; better suited for multi-agent collaboration
- **Temporal**: Production-grade but requires significant infrastructure setup
- **Prefect/Airflow**: Better for data pipelines; less suited for LLM-driven workflows

## Pipeline Architecture

### Current Flow (Streamlit-based)
```
User Input → Translation → Interpretation → [Human Confirmation] → Validation → Data Fetch → Feature Engineering → Pre-backtest QA → Backtest → Metrics → Explanation
```

### Proposed Agentic Flow (LangGraph-based)
```
[State Initialization]
    ↓
[Translation Node] → [Error Handler] → [Retry Logic]
    ↓
[Interpretation Node] → [Human-in-the-Loop Checkpoint]
    ↓
[User Confirmation] → [Conditional: Proceed or Reset]
    ↓
[Validation Node] → [Conditional: Pass or Fail]
    ↓
[Data Fetching Node] → [Error Handler]
    ↓
[Feature Engineering Node]
    ↓
[Pre-backtest QA Node] → [Conditional: Pass or Fail]
    ↓
[Backtest Execution Node]
    ↓
[Metrics Computation Node]
    ↓
[Explanation Node]
    ↓
[Results Aggregation Node]
    ↓
[State Persistence]
```

## Detailed Pipeline Steps

### Step 1: State Initialization
**Node Type**: Initialization
**Purpose**: Set up the pipeline state with user input and configuration
**Inputs**:
- User natural language strategy description
- Selected LLM model (gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
- Optional: User preferences, session ID

**State Variables Created**:
```python
{
    "user_text": str,
    "model": str,
    "spec": StrategySpec | None,
    "interpretation": str | None,
    "confirmed": bool,
    "validation_result": ValidationResult | None,
    "data": pd.DataFrame | None,
    "backtest_results": pd.DataFrame | None,
    "metrics": Dict[str, float] | None,
    "explanation": str | None,
    "trades": List[Trade] | None,
    "errors": List[str],
    "warnings": List[str],
    "metrics_log": List[Dict],  # For tracking LLM calls
    "session_id": str,
    "timestamp": datetime
}
```

**Output**: Initialized state object
**Error Handling**: Validate inputs, ensure API key is set

---

### Step 2: Translation Node
**Node Type**: LLM Call Node
**Purpose**: Convert natural language to StrategySpec JSON
**Function**: `translate_to_spec(user_text, model)`
**Dependencies**: OpenAI API, existing `llm.translator` module

**State Updates**:
- `state["spec"]` = parsed StrategySpec
- `state["metrics_log"]` += translation metrics (tokens, cost, latency)

**Error Handling**:
- Retry on API failures (max 3 retries with exponential backoff)
- Validate JSON structure
- Handle parsing errors gracefully
- Log errors to `state["errors"]`

**Conditional Routing**:
- Success → Proceed to Interpretation Node
- Failure → Error Recovery Node → Retry or User Notification

---

### Step 3: Interpretation Node
**Node Type**: LLM Call Node
**Purpose**: Generate human-readable interpretation explanation
**Function**: `explain_interpretation(user_text, spec, model)`
**Dependencies**: OpenAI API, existing `llm.interpreter` module

**State Updates**:
- `state["interpretation"]` = explanation text
- `state["metrics_log"]` += interpretation metrics

**Error Handling**:
- Non-critical: If interpretation fails, continue with default message
- Log warnings to `state["warnings"]`

**Conditional Routing**:
- Always proceed to Human-in-the-Loop Checkpoint

---

### Step 4: Human-in-the-Loop Checkpoint
**Node Type**: Checkpoint (LangGraph special node)
**Purpose**: Pause execution and wait for user confirmation
**Dependencies**: Streamlit UI or API endpoint

**State Updates**:
- `state["confirmed"]` = False (initially)
- State is persisted to allow resumption

**User Interaction**:
- Display interpretation and default assumptions
- Show parsed strategy specification
- Wait for user to click "Confirm & Proceed" or "Edit & Retry"

**Conditional Routing**:
- User confirms → Proceed to Validation Node
- User requests edit → Reset to Translation Node with new input
- Timeout (optional) → Notify user, pause execution

**Implementation Notes**:
- Use LangGraph's `interrupt_before` or `interrupt_after` decorator
- State must be serializable for persistence
- Can integrate with Streamlit's session state or separate API

---

### Step 5: Validation Node
**Node Type**: Validation Node
**Purpose**: Validate strategy structure and logic
**Function**: `validate_spec(spec)`
**Dependencies**: Existing `core.validator` module

**State Updates**:
- `state["validation_result"]` = ValidationResult
- `state["errors"]` += validation errors
- `state["warnings"]` += validation warnings

**Error Handling**:
- Critical errors block progression
- Warnings are logged but don't block

**Conditional Routing**:
- No errors → Proceed to Data Fetching Node
- Has errors → Error Notification Node → Stop or allow retry

---

### Step 6: Data Fetching Node
**Node Type**: Data Processing Node
**Purpose**: Download price data from yfinance
**Function**: `load_price_data(spec)`
**Dependencies**: yfinance, existing `core.data` module

**State Updates**:
- `state["data"]` = DataFrame with OHLCV data

**Error Handling**:
- Retry on network failures (max 3 retries)
- Handle missing data gracefully
- Validate date range availability
- Log errors to `state["errors"]`

**Conditional Routing**:
- Success → Proceed to Feature Engineering Node
- Failure → Error Recovery Node → Retry or User Notification

---

### Step 7: Feature Engineering Node
**Node Type**: Data Processing Node
**Purpose**: Compute moving averages, volatility, and other features
**Function**: `add_features(df, spec)`
**Dependencies**: pandas, numpy, existing `core.data` module

**State Updates**:
- `state["data"]` = DataFrame with added features (MAs, RV, medians)

**Error Handling**:
- Handle insufficient data for lookback windows
- Validate feature computation
- Log warnings for edge cases

**Conditional Routing**:
- Always proceed to Pre-backtest QA Node

---

### Step 8: Pre-backtest QA Node
**Node Type**: Validation Node
**Purpose**: Data-dependent validation before backtest
**Function**: `validate_with_data(spec, df)`
**Dependencies**: Existing `core.validator` module

**State Updates**:
- `state["validation_result"]` = updated ValidationResult
- `state["warnings"]` += data-dependent warnings

**Error Handling**:
- Critical errors (no data, impossible conditions) block progression
- Warnings (zero trades likely) are logged but don't block

**Conditional Routing**:
- No critical errors → Proceed to Backtest Execution Node
- Has critical errors → Error Notification Node → Stop or allow retry

---

### Step 9: Backtest Execution Node
**Node Type**: Computation Node
**Purpose**: Execute the backtest logic
**Function**: `run_backtest(df, spec)`
**Dependencies**: Existing `core.backtester` module

**State Updates**:
- `state["backtest_results"]` = DataFrame with positions, returns, equity curve

**Error Handling**:
- Handle edge cases (no trades, all trades, etc.)
- Validate backtest output structure
- Log computational errors

**Conditional Routing**:
- Always proceed to Metrics Computation Node

---

### Step 10: Metrics Computation Node
**Node Type**: Computation Node
**Purpose**: Calculate performance metrics
**Function**: `compute_basic_metrics(results_df)`
**Dependencies**: Existing `core.metrics` module

**State Updates**:
- `state["metrics"]` = Dict with CAGR, drawdown, Sharpe, trade count

**Error Handling**:
- Handle division by zero, NaN values
- Validate metric calculations

**Conditional Routing**:
- Always proceed to Trade Extraction Node (parallel or sequential)

---

### Step 11: Trade Extraction Node (Parallel)
**Node Type**: Computation Node
**Purpose**: Extract detailed trade-by-trade information
**Function**: `extract_trades(df, spec)`
**Dependencies**: Existing `core.backtester` module

**State Updates**:
- `state["trades"]` = List[Trade] objects

**Error Handling**:
- Handle edge cases (no trades, open positions)

**Conditional Routing**:
- Always proceed to Explanation Node

---

### Step 12: Explanation Node
**Node Type**: LLM Call Node
**Purpose**: Generate human-readable explanation of results
**Function**: `summarize_results(spec, metrics, model)`
**Dependencies**: OpenAI API, existing `llm.explainer` module

**State Updates**:
- `state["explanation"]` = explanation text
- `state["metrics_log"]` += explanation metrics

**Error Handling**:
- Non-critical: If explanation fails, use default summary
- Log warnings

**Conditional Routing**:
- Always proceed to Results Aggregation Node

---

### Step 13: Results Aggregation Node
**Node Type**: Finalization Node
**Purpose**: Compile all results and prepare for display
**Function**: Aggregate state into final output format

**State Updates**:
- Final state ready for consumption

**Output Format**:
```python
{
    "strategy_spec": StrategySpec.to_dict(),
    "interpretation": str,
    "validation": {
        "errors": List[str],
        "warnings": List[str]
    },
    "backtest_results": {
        "equity_curve": pd.DataFrame,
        "metrics": Dict[str, float],
        "trades": List[Trade],
        "explanation": str
    },
    "llm_metrics": {
        "total_tokens": int,
        "total_cost": float,
        "total_latency": float,
        "calls": List[Dict]
    },
    "session_info": {
        "session_id": str,
        "timestamp": datetime,
        "model_used": str
    }
}
```

**Conditional Routing**:
- Always proceed to State Persistence Node

---

### Step 14: State Persistence Node
**Node Type**: Persistence Node
**Purpose**: Save state and metrics for analysis
**Function**: Save to CSV, database, or file system

**Actions**:
- Append to `metrics_log.csv` (existing format)
- Optionally save full state snapshot
- Update session tracking

**Conditional Routing**:
- End of pipeline

---

## Error Recovery & Retry Logic

### Retry Strategy
1. **LLM API Calls**: 
   - Max 3 retries with exponential backoff (1s, 2s, 4s)
   - Retry on: rate limits, network errors, timeouts
   - Don't retry on: invalid API key, malformed requests

2. **Data Fetching**:
   - Max 3 retries with exponential backoff
   - Retry on: network errors, temporary yfinance failures
   - Don't retry on: invalid ticker, date range issues

3. **Validation Failures**:
   - No automatic retry (requires user input)
   - Route to error notification

### Error Recovery Nodes
- **Error Notification Node**: Display errors to user, allow manual retry
- **Error Recovery Node**: Attempt automatic recovery (retry, fallback)
- **Graceful Degradation**: Continue with partial results when non-critical steps fail

## State Management

### State Schema
```python
from typing import TypedDict, List, Dict, Optional
from datetime import datetime
import pandas as pd

class PipelineState(TypedDict):
    # Input
    user_text: str
    model: str
    session_id: str
    timestamp: datetime
    
    # Intermediate Results
    spec: Optional[StrategySpec]
    interpretation: Optional[str]
    confirmed: bool
    validation_result: Optional[ValidationResult]
    data: Optional[pd.DataFrame]
    backtest_results: Optional[pd.DataFrame]
    metrics: Optional[Dict[str, float]]
    explanation: Optional[str]
    trades: Optional[List[Trade]]
    
    # Metadata
    errors: List[str]
    warnings: List[str]
    metrics_log: List[Dict]
    current_step: str
    retry_count: Dict[str, int]  # Track retries per node
```

### State Persistence
- Use LangGraph's built-in checkpointing
- Serialize state to JSON (handle DataFrame serialization separately)
- Store checkpoints in file system or database
- Enable resumption from any checkpoint

## Integration Points

### With Existing Codebase
1. **LLM Modules**: Keep existing `llm/translator.py`, `llm/interpreter.py`, `llm/explainer.py` as-is
2. **Core Modules**: Keep existing `core/` modules as-is
3. **Metrics Tracking**: Integrate with existing `utils/metrics_tracker.py`
4. **Streamlit UI**: Can either:
   - Replace Streamlit with LangGraph + API
   - Keep Streamlit as frontend, use LangGraph as backend
   - Hybrid: Streamlit triggers LangGraph pipeline

### With External Services
1. **OpenAI API**: Direct integration (already in place)
2. **yfinance**: Direct integration (already in place)
3. **LangSmith** (optional): For observability and monitoring
4. **Database** (optional): For state persistence and history

## Implementation Phases

### Phase 1: Core Pipeline (Weeks 1-2)
- Set up LangGraph project structure
- Implement Steps 1-5 (Initialization through Validation)
- Basic error handling
- Unit tests for each node

### Phase 2: Data & Backtest (Weeks 3-4)
- Implement Steps 6-9 (Data Fetching through Backtest)
- Integrate with existing modules
- Add retry logic
- Integration tests

### Phase 3: Results & Persistence (Weeks 5-6)
- Implement Steps 10-14 (Metrics through Persistence)
- State persistence
- Metrics aggregation
- End-to-end tests

### Phase 4: Human-in-the-Loop (Week 7)
- Implement checkpoint mechanism
- Integrate with Streamlit or API
- Test user interaction flows

### Phase 5: Production Readiness (Week 8)
- Error recovery improvements
- Performance optimization
- Documentation
- Deployment setup

## Benefits of Agentic Pipeline

1. **Better Error Handling**: Automatic retries, graceful degradation
2. **State Management**: Persistent state across sessions
3. **Observability**: Track each step, debug issues easily
4. **Scalability**: Can handle multiple concurrent requests
5. **Flexibility**: Easy to add new steps or modify flow
6. **Testing**: Each node can be tested independently
7. **Cost Tracking**: Better visibility into LLM usage
8. **Resumability**: Can resume from any checkpoint

## Migration Strategy

1. **Parallel Development**: Build LangGraph pipeline alongside existing Streamlit app
2. **Feature Parity**: Ensure all existing features work in new pipeline
3. **Gradual Migration**: Start with new features in LangGraph, migrate existing ones
4. **A/B Testing**: Run both systems in parallel, compare results
5. **Full Migration**: Once stable, replace Streamlit workflow with LangGraph

## Dependencies to Add

```txt
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langsmith>=0.1.0  # Optional, for observability
```

## Example LangGraph Code Structure

```python
# pipeline/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from .nodes import (
    initialize_state,
    translate_strategy,
    generate_interpretation,
    validate_strategy,
    fetch_data,
    add_features,
    pre_backtest_qa,
    run_backtest,
    compute_metrics,
    extract_trades,
    generate_explanation,
    aggregate_results,
    persist_state
)
from .checkpoints import human_confirmation_checkpoint

def create_pipeline():
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("translate", translate_strategy)
    workflow.add_node("interpret", generate_interpretation)
    workflow.add_node("validate", validate_strategy)
    workflow.add_node("fetch_data", fetch_data)
    workflow.add_node("add_features", add_features)
    workflow.add_node("pre_qa", pre_backtest_qa)
    workflow.add_node("backtest", run_backtest)
    workflow.add_node("metrics", compute_metrics)
    workflow.add_node("trades", extract_trades)
    workflow.add_node("explain", generate_explanation)
    workflow.add_node("aggregate", aggregate_results)
    workflow.add_node("persist", persist_state)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "translate")
    workflow.add_edge("translate", "interpret")
    workflow.add_conditional_edges(
        "interpret",
        should_confirm,
        {
            "confirm": "validate",
            "reset": "translate"
        }
    )
    workflow.add_conditional_edges(
        "validate",
        check_validation,
        {
            "pass": "fetch_data",
            "fail": END
        }
    )
    # ... continue with rest of edges
    
    # Add checkpoint for human-in-the-loop
    workflow.add_edge("interpret", human_confirmation_checkpoint)
    
    return workflow.compile(checkpointer=...)
```

## Conclusion

LangGraph provides the best foundation for converting this project into a robust, scalable agentic pipeline. The proposed architecture maintains all existing functionality while adding better error handling, state management, and observability. The phased implementation approach ensures a smooth migration without disrupting the current system.

