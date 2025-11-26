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
  
- **`explainer.py`**: Generates human-readable explanations of backtest results
  - Analyzes performance metrics (CAGR, drawdown, Sharpe ratio, trade count)
  - Provides concise explanations for retail traders

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

#### 3. **Application Layer** (`app.py`)
- Streamlit web interface
- Orchestrates the complete pipeline:
  1. Natural language input → Strategy specification
  2. Data fetching and feature engineering
  3. Backtest execution
  4. Metrics computation
  5. Visualization
  6. LLM-generated explanation

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
- JSON-structured responses for strategy translation
- Natural language explanations of results

### ✅ Logging & Error Handling
- Comprehensive logging at each pipeline stage
- Error handling with user-friendly messages
- Success/failure tracking for all operations

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI API (GPT-4o-mini)
- **Data**: yfinance, pandas, numpy
- **Visualization**: matplotlib
- **Configuration**: python-dotenv (for API key management)

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
User Input (Natural Language)
    ↓
LLM Translation → StrategySpec (JSON)
    ↓
Data Fetching (yfinance)
    ↓
Feature Engineering (MAs, Volatility)
    ↓
Backtest Execution
    ↓
Metrics Computation
    ↓
Visualization + LLM Explanation
```

## Configuration

- API keys managed via `.env` file
- Logging configured for pipeline monitoring
- Error handling at each stage with graceful degradation

## Current Status

All core functionality is implemented and operational:
- ✅ Natural language to strategy translation
- ✅ Data fetching and feature engineering
- ✅ Backtesting engine
- ✅ Performance metrics
- ✅ Visualization
- ✅ LLM explanations
- ✅ Logging and error handling

