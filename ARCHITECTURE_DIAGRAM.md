# Backtest Chat Copilot - Architecture Diagram

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Streamlit Web App]
        UI --> |User Input| Pipeline[Agentic Pipeline]
        Pipeline --> |Results| UI
    end
    
    subgraph "Agentic Pipeline Layer - LangGraph"
        Pipeline --> |State Management| State[Pipeline State]
        Pipeline --> |Checkpointing| Checkpoint[Memory Checkpointer]
        
        subgraph "Pipeline Nodes"
            N1[Initialize Node]
            N2[Translate Node]
            N3[Interpret Node]
            N4[Validate Node]
            N5[Fetch Data Node]
            N6[Add Features Node]
            N7[Pre-QA Node]
            N8[Backtest Node]
            N9[Metrics Node]
            N10[Trades Node]
            N11[Explain Node]
            N12[Aggregate Node]
            N13[Persist Node]
        end
        
        Pipeline --> N1
        N1 --> N2
        N2 --> N3
        N3 --> |Human-in-the-Loop| Checkpoint
        Checkpoint --> |User Confirms| N4
        N4 --> |Validation Pass| N5
        N4 --> |Validation Fail| END1[End]
        N5 --> N6
        N6 --> N7
        N7 --> |Data Validation Pass| N8
        N7 --> |Data Validation Fail| END2[End]
        N8 --> N9
        N9 --> N10
        N10 --> N11
        N11 --> N12
        N12 --> N13
        
        Pipeline --> |Error Handling| Retry[Retry Logic with Backoff]
    end
    
    subgraph "LLM Services Layer"
        N2 --> |API Call| LLM1[OpenAI API<br/>Translation]
        N3 --> |API Call| LLM2[OpenAI API<br/>Interpretation]
        N11 --> |API Call| LLM3[OpenAI API<br/>Explanation]
        
        LLM1 --> |Metrics| MetricsLog[Metrics Tracker]
        LLM2 --> |Metrics| MetricsLog
        LLM3 --> |Metrics| MetricsLog
    end
    
    subgraph "Core Backtesting Engine"
        N5 --> |Fetch| DataSource[yfinance API]
        DataSource --> |OHLCV Data| N6
        N6 --> |Feature Engineering| N8
        N8 --> |Backtest Logic| N9
        N8 --> |Trade Extraction| N10
    end
    
    subgraph "Validation & Quality Assurance"
        N4 --> |Structure Check| Validator1[Strategy Validator]
        N7 --> |Data Check| Validator2[Data Validator]
    end
    
    subgraph "State & Persistence"
        State --> |Serialization| Checkpoint
        MetricsLog --> |CSV Export| CSVFile[metrics_log.csv]
        N13 --> |State Persistence| Checkpoint
    end
    
    style Pipeline fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style Checkpoint fill:#50C878,stroke:#2E7D4E,stroke-width:2px,color:#fff
    style Retry fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px,color:#fff
    style UI fill:#9B59B6,stroke:#6C3483,stroke-width:2px,color:#fff
```

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A[Natural Language<br/>Strategy Description] --> B[Model Selection<br/>gpt-4o-mini/gpt-4o]
    end
    
    subgraph "Agentic Pipeline - LangGraph"
        B --> C[State Initialization]
        C --> D[Translation Node]
        D --> E[Interpretation Node]
        E --> F{User<br/>Confirmation?}
        F -->|No| G[Checkpoint<br/>Wait]
        F -->|Yes| H[Validation Node]
        H --> I{Validation<br/>Pass?}
        I -->|No| J[Error Display]
        I -->|Yes| K[Data Fetching Node]
        K --> L[Feature Engineering Node]
        L --> M[Pre-backtest QA Node]
        M --> N{Data Validation<br/>Pass?}
        N -->|No| J
        N -->|Yes| O[Backtest Node]
        O --> P[Metrics Node]
        P --> Q[Trades Node]
        Q --> R[Explanation Node]
        R --> S[Results Aggregation]
        S --> T[State Persistence]
    end
    
    subgraph "External Services"
        D -.->|API Call| U[OpenAI API]
        E -.->|API Call| U
        R -.->|API Call| U
        K -.->|Data Fetch| V[yfinance]
    end
    
    subgraph "Output Layer"
        T --> W[Equity Curve Plot]
        T --> X[Performance Metrics]
        T --> Y[Trade-by-Trade Details]
        T --> Z[LLM Explanation]
    end
    
    style F fill:#FFD700,stroke:#FFA500,stroke-width:2px
    style I fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px
    style N fill:#FF6B6B,stroke:#C92A2A,stroke-width:2px
    style G fill:#50C878,stroke:#2E7D4E,stroke-width:2px
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Pipeline
    participant Checkpoint
    participant LLM
    participant DataSource
    participant Backtester
    
    User->>Streamlit: Enter Strategy Description
    Streamlit->>Pipeline: run_pipeline(user_text, model)
    Pipeline->>Pipeline: Initialize State
    Pipeline->>LLM: Translate to StrategySpec
    LLM-->>Pipeline: StrategySpec JSON
    Pipeline->>LLM: Generate Interpretation
    LLM-->>Pipeline: Interpretation Text
    Pipeline->>Checkpoint: Save State (Pause)
    Checkpoint-->>Streamlit: State at Checkpoint
    Streamlit->>User: Show Interpretation & Spec
    User->>Streamlit: Click "Confirm & Proceed"
    Streamlit->>Pipeline: resume_pipeline(session_id, confirmed=True)
    Pipeline->>Checkpoint: Load State
    Pipeline->>Pipeline: Validate Strategy
    Pipeline->>DataSource: Fetch Price Data
    DataSource-->>Pipeline: OHLCV DataFrame
    Pipeline->>Pipeline: Add Features (MAs, Volatility)
    Pipeline->>Pipeline: Pre-backtest QA
    Pipeline->>Backtester: Run Backtest
    Backtester-->>Pipeline: Results DataFrame
    Pipeline->>Pipeline: Compute Metrics
    Pipeline->>LLM: Generate Explanation
    LLM-->>Pipeline: Explanation Text
    Pipeline->>Checkpoint: Save Final State
    Pipeline-->>Streamlit: Complete State with Results
    Streamlit->>User: Display Results (Charts, Metrics, Trades)
```

## Key Benefits of This Architecture

### 1. **Agentic Pipeline Benefits**

#### ✅ **State Management & Persistence**
- **Benefit**: Complete state tracking across all pipeline stages
- **Impact**: Can resume from any point, debug issues easily, track execution history
- **Implementation**: LangGraph's built-in state management with checkpointing

#### ✅ **Human-in-the-Loop Integration**
- **Benefit**: User confirmation before expensive operations (data fetching, backtesting)
- **Impact**: Prevents wasted API calls and computation on incorrect interpretations
- **Implementation**: Checkpoint mechanism pauses before validation, waits for user confirmation

#### ✅ **Error Recovery & Retry Logic**
- **Benefit**: Automatic retries with exponential backoff for transient failures
- **Impact**: Higher reliability, better user experience, reduced manual intervention
- **Implementation**: Decorator-based retry logic for LLM calls and data fetching

#### ✅ **Conditional Routing**
- **Benefit**: Smart pipeline flow based on validation results
- **Impact**: Early failure detection, prevents unnecessary computation
- **Implementation**: Conditional edges in LangGraph based on validation status

### 2. **Modularity & Maintainability**

#### ✅ **Separation of Concerns**
- **Benefit**: Clear boundaries between UI, pipeline, LLM, and core logic
- **Impact**: Easy to modify, test, and extend individual components
- **Structure**: 
  - `app.py`: UI layer only
  - `pipeline/`: Agentic orchestration
  - `llm/`: LLM interactions
  - `core/`: Business logic

#### ✅ **Reusable Components**
- **Benefit**: Nodes can be reused, tested independently, or composed differently
- **Impact**: Faster development, easier testing, flexible architecture

### 3. **Observability & Debugging**

#### ✅ **Comprehensive Logging**
- **Benefit**: Track execution at every step with detailed logs
- **Impact**: Easy debugging, performance monitoring, cost tracking
- **Implementation**: Logging at node level, state tracking, metrics collection

#### ✅ **Metrics Tracking**
- **Benefit**: Track LLM usage (tokens, costs, latency) for all operations
- **Impact**: Cost optimization, performance analysis, usage monitoring
- **Implementation**: Automatic metrics logging to CSV

### 4. **Scalability & Performance**

#### ✅ **Efficient State Handling**
- **Benefit**: State only serialized when needed (checkpoints), DataFrames handled separately
- **Impact**: Faster execution, lower memory usage
- **Implementation**: Stream-based state capture, checkpoint optimization

#### ✅ **Parallel-Ready Architecture**
- **Benefit**: Nodes can be easily parallelized in future (e.g., metrics + trades)
- **Impact**: Potential for faster execution with concurrent processing

### 5. **User Experience**

#### ✅ **Progressive Disclosure**
- **Benefit**: Show interpretation first, get confirmation, then run expensive operations
- **Impact**: Better user trust, fewer errors, clearer expectations

#### ✅ **Error Resilience**
- **Benefit**: Graceful degradation - non-critical failures don't stop the pipeline
- **Impact**: Better user experience, partial results when possible

### 6. **Cost Optimization**

#### ✅ **Selective LLM Usage**
- **Benefit**: Only use LLM when needed, with model selection for cost/quality tradeoff
- **Impact**: Lower costs, faster execution with cheaper models when appropriate

#### ✅ **Retry Logic**
- **Benefit**: Automatic retry on transient failures reduces manual retries
- **Impact**: Lower operational overhead, better success rates

## Architecture Comparison

### Before (Sequential Streamlit)
```
User Input → Translation → Interpretation → [Manual Confirmation] 
→ Validation → Data Fetch → Backtest → Results
```
**Issues:**
- No state persistence
- No error recovery
- No retry logic
- Tightly coupled components
- Difficult to debug

### After (Agentic Pipeline)
```
User Input → [Agentic Pipeline with State Management]
→ Checkpoint → User Confirmation → Resume
→ [Error Recovery & Retries] → Results
```
**Benefits:**
- ✅ State persistence
- ✅ Automatic error recovery
- ✅ Retry logic
- ✅ Modular components
- ✅ Easy debugging
- ✅ Scalable architecture

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Streamlit | Web interface |
| **Orchestration** | LangGraph | Agentic pipeline |
| **State** | MemorySaver | Checkpointing |
| **LLM** | OpenAI API | Translation, interpretation, explanation |
| **Data** | yfinance | Market data |
| **Computation** | pandas, numpy | Data processing & backtesting |
| **Visualization** | matplotlib | Equity curves |
| **Metrics** | CSV | Logging & analysis |

## Key Design Decisions

1. **LangGraph over LangChain**: Better state management and human-in-the-loop support
2. **Checkpointing**: Enables resumability and user confirmation
3. **Stream-based execution**: Captures state even when checkpointing fails
4. **Shared checkpointer**: Singleton pattern ensures state persistence across instances
5. **Modular nodes**: Each node is independent and testable
6. **Error handling**: Retry logic for transient failures, graceful degradation for non-critical

## Future Enhancements

- **Parallel execution**: Run metrics and trades nodes concurrently
- **Database persistence**: Replace MemorySaver with database-backed checkpointer
- **Caching**: Cache LLM responses for similar strategies
- **Batch processing**: Process multiple strategies in parallel
- **Advanced analytics**: Add more sophisticated performance metrics
- **Multi-asset support**: Extend to portfolio strategies

