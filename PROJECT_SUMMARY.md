# Backtest Chat Copilot - Project Summary

## Overview
A Streamlit-based web application that uses LLMs to translate natural language trading strategy descriptions into executable backtests. The system enables users to describe single-asset, long-only trading strategies in plain English and automatically generates backtest results with performance metrics and explanations.

## Architecture

### Core Components

#### 1. **LLM Module** (`llm/`)
- **`translator.py`**: Converts natural language strategy descriptions into structured `StrategySpec` JSON using OpenAI's GPT models
  - Uses system prompts to enforce strategy constraints (single-asset, long-only)
  - Supports moving average crossovers and volatility filters
  - Returns validated JSON strategy specifications
  - Token usage and cost tracking integrated
  
- **`interpreter.py`**: Explain-back NL interpreter with human-in-the-loop confirmation
  - Explains how the user's strategy was interpreted
  - Highlights critical ambiguities that would prevent execution
  - Shows default assumptions and technical values side-by-side
  - Requires explicit user confirmation before proceeding with backtest
  
- **`explainer.py`**: Generates human-readable explanations of backtest results
  - Analyzes performance metrics (CAGR, drawdown, Sharpe ratio, trade count)
  - Provides concise explanations for retail traders
  - Token usage and cost tracking integrated

#### 2. **Core Backtesting Engine** (`core/`)
- **`strategy_spec.py`**: Defines data structures for trading strategies
  - `StrategySpec`: Main specification class (ticker, dates, entry/exit rules, metrics)
  - `CrossoverRule`: Moving average crossover rules (fast MA vs slow MA)
  - `VolFilterRule`: Volatility filter rules (realized vol vs 1-year median)
  
- **`data.py`**: Data fetching and feature engineering
  - Downloads OHLCV data via yfinance
  - Computes moving averages (configurable windows)
  - Calculates realized volatility and 1-year rolling medians
  
- **`backtester.py`**: Executes the backtest logic
  - Long-only position management (0 or 1.0 positions)
  - Entry: All entry rules must be satisfied
  - Exit: Any exit rule triggers position closure
  - Generates equity curve and strategy returns
  
- **`metrics.py`**: Performance metric calculations
  - CAGR (Compound Annual Growth Rate)
  - Maximum drawdown
  - Sharpe ratio (annualized)
  - Trade count
  
- **`plotting.py`**: Visualization
  - Equity curve plotting with matplotlib

- **`validator.py`**: Strategy validation and pre-backtest QA
  - `validate_spec()`: Validates strategy structure and logic (before data loading)
    - Date range validation
    - Rule parameter validation
    - Logic conflict detection
    - Reasonableness checks
  - `validate_with_data()`: Data-dependent validation (after data loading)
    - Data availability checks
    - Lookback window sufficiency
    - Rule feasibility testing on historical data
    - Zero-trade warnings

#### 3. **Utilities** (`utils/`)
- **`metrics_tracker.py`**: Quantitative metrics tracking
  - Logs token usage (input/output/total) for each LLM call
  - Calculates costs based on model pricing
  - Tracks latency and success/failure rates
  - Exports to CSV for analysis

#### 4. **Analysis** (`analysis/`)
- **`analyze_metrics.py`**: Metrics analysis script
  - Aggregates metrics by task type and model
  - Generates summary statistics (success rates, avg tokens, costs, latencies)
  - Outputs comparison tables and overall totals

#### 5. **Application Layer** (`app.py`)
- Streamlit web interface with model selection dropdown
- Orchestrates the complete pipeline:
  1. Model selection (gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
  2. Natural language input → Strategy specification
  3. **Interpretation explanation with human-in-the-loop confirmation**
  4. Strategy validation (structure and logic checks)
  5. Data fetching and feature engineering
  6. Pre-backtest QA (data-dependent validation)
  7. Backtest execution
  8. Metrics computation
  9. Visualization
  10. LLM-generated explanation

## Key Features Implemented

### ✅ Natural Language Processing
- Converts plain English strategy descriptions to structured JSON
- Validates and enforces strategy constraints
- Handles date parsing, ticker symbols, and rule specifications

### ✅ Data Management
- Automatic data fetching from Yahoo Finance
- Feature engineering (moving averages, volatility calculations)
- Date range validation and error handling

### ✅ Backtesting Engine
- Long-only position management
- Rule-based entry/exit logic
- Close-to-close return calculation
- Equity curve generation

### ✅ Performance Analytics
- Standard performance metrics (CAGR, Sharpe, drawdown)
- Trade counting
- Visual equity curve display

### ✅ LLM Integration
- OpenAI API integration with environment variable configuration
- **Model selection dropdown** (gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
- JSON-structured responses for strategy translation
- Natural language explanations of results
- **Quantitative metrics tracking** (tokens, costs, latency)

### ✅ Explain-Back NL Interpreter
- **Human-in-the-loop confirmation** before backtest execution
- Explains how strategy was interpreted
- Shows default assumptions and technical values side-by-side
- Highlights only critical ambiguities that would prevent execution
- Requires explicit user confirmation to proceed

### ✅ Strategy Validation & QA
- **Pre-translation validation**: Structure and logic checks
- **Pre-backtest QA**: Data-dependent validation
- Detects conflicting rules, insufficient data, unreachable conditions
- Warns about potential issues (zero trades, long lookback windows)

### ✅ Logging & Error Handling
- Comprehensive logging at each pipeline stage
- Error handling with user-friendly messages
- Success/failure tracking for all operations
- **Metrics logging to CSV** for quantitative analysis

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI API (GPT-4o-mini, GPT-4o, GPT-4-turbo, GPT-3.5-turbo) with model selection
- **Data**: yfinance, pandas, numpy
- **Visualization**: matplotlib
- **Configuration**: python-dotenv (for API key management)
- **Metrics**: CSV logging with cost and token tracking

## Supported Strategy Types

### Entry/Exit Rules
1. **Moving Average Crossovers**
   - Fast MA crosses above/below slow MA
   - Configurable window sizes

2. **Volatility Filters**
   - Realized volatility vs 1-year median
   - Above/below threshold comparisons

### Constraints
- Single-asset strategies only
- Long-only (no shorting, no leverage)
- Daily timeframe (close-to-close)
- Standard date range support

## Pipeline Flow

```
User Input (Natural Language) + Model Selection
    ↓
LLM Translation → StrategySpec (JSON)
    ↓
Interpretation Explanation + Default Assumptions Display
    ↓
User Confirmation (Human-in-the-Loop)
    ↓
Strategy Validation (Structure & Logic)
    ↓
Data Fetching (yfinance)
    ↓
Feature Engineering (MAs, Volatility)
    ↓
Pre-backtest QA (Data-Dependent Validation)
    ↓
Backtest Execution
    ↓
Metrics Computation
    ↓
Visualization + LLM Explanation
```

**Note**: All LLM calls (translation, interpretation, explanation) are logged with token usage, costs, and latency metrics.

## Configuration

- API keys managed via `.env` file
- Logging configured for pipeline monitoring
- Error handling at each stage with graceful degradation

## Current Status

All core functionality is implemented and operational:
- ✅ Natural language to strategy translation
- ✅ **Explain-back NL interpreter with human-in-the-loop confirmation**
- ✅ **Model selection dropdown (multiple OpenAI models)**
- ✅ **Strategy validation and pre-backtest QA**
- ✅ Data fetching and feature engineering
- ✅ Backtesting engine
- ✅ Performance metrics
- ✅ Visualization
- ✅ LLM explanations
- ✅ **Quantitative metrics tracking (tokens, costs, latency)**
- ✅ **Metrics analysis script**
- ✅ Logging and error handling

## Quantitative Analysis

The system tracks comprehensive metrics for all LLM operations:
- **Token usage**: Input, output, and total tokens per call
- **Cost tracking**: Automatic cost calculation based on model pricing
- **Latency**: Response time for each LLM call
- **Success rates**: Track failures and retries
- **Analysis**: Run `python analysis/analyze_metrics.py` to generate summary statistics

Metrics are logged to `metrics_log.csv` and can be analyzed to compare model performance, costs, and efficiency.

