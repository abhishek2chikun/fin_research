# Financial Agent System

## Overview

The Financial Agent System is a comprehensive platform for AI-driven investment analysis that leverages specialized AI agents to analyze stocks and provide investment insights. The system processes historical financial data (both fundamental and OHLCV) to generate investment signals and recommendations based on different investment philosophies. The system supports both LangChain and Agno frameworks, with no requirement to convert existing LangChain agents to Agno unless specifically needed.

## Key Features

- **Multiple Investment Strategy Agents**: Each agent implements a different investment philosophy (Ben Graham, Warren Buffett, etc.)
- **Data Processing**: Efficient handling of historical and real-time financial data
- **Technical Analysis**: Comprehensive tools for technical indicator calculation and analysis
- **Interactive UI**: Streamlit-based interface with visualizations and reports
- **Multi-LLM Support**: Integration with multiple LLM providers (OpenAI, Anthropic, DeepSeek, Gemini, Groq, LMStudio)

## Architecture

The system is built with a modular architecture consisting of the following layers:

### 1. Data Layer

- **Fundamental Data**: JSON files containing company financial metrics, balance sheets, income statements, etc.
- **OHLCV Data**: CSV files with Open, High, Low, Close, Volume data for stocks
- **Data Models**: Pydantic models for structured data representation and validation

### 2. Agent Layer

- **Investment Strategy Agents**: Specialized agents named after famous investors
  - Ben Graham (Value investing)
  - Warren Buffett
  - Charlie Munger
  - Bill Ackman
  - Cathie Wood
  - Phil Fisher
  - Stanley Druckenmiller
- **Support Agents**:
  - Portfolio Manager
  - Risk Manager
  - Sentiment Analysis
  - Technical Analysis
  - Valuation
  - Fundamentals

> **Note**: Agents can be implemented using either LangChain or Agno frameworks. There is no requirement to convert existing LangChain agents to Agno unless specific Agno features are needed.

### 3. Tools Layer

- Financial data retrieval tools
- Market data analysis tools
- Intraday data processing
- Technical indicator calculation

### 4. UI Layer

- Streamlit-based web interface
- Multi-page application structure
- Interactive visualizations and reports

### 5. LLM Integration

- Support for multiple LLM providers:
  - OpenAI
  - Anthropic
  - DeepSeek
  - Gemini
  - Groq
  - LMStudio (local models)

## Directory Structure

```
fin_agent/
├── agents/                   # Investment strategy agents
│   ├── base_agent.py         # Base agent architecture
│   ├── ben_graham.py         # Original Ben Graham agent (LangChain)
│   ├── ben_graham_agno.py    # Ben Graham agent (Agno)
│   ├── bill_ackman.py        # Bill Ackman agent
│   └── ...                   # Other investment strategy agents
├── data/                     # Data models and processing
│   ├── fundamental_data.py   # Fundamental data models
│   └── ohlcv.py              # OHLCV data models
├── graph/                    # Agent state and visualization
│   └── state.py              # Agent state management
├── Historical_data/          # Historical financial data (CSV files)
│   └── ...                   # Stock data files
├── llm/                      # LLM integration
│   └── providers.py          # Multi-provider LLM integration
├── tools/                    # Financial analysis tools
│   └── intraday_data.py      # Intraday data processing tools
├── UI/                       # Streamlit UI
│   ├── app.py                # Main Streamlit application
│   └── pages/                # Additional UI pages
├── utils/                    # Utility functions
│   ├── progress.py           # Progress tracking
│   └── llm.py                # LLM utility functions
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Technical Implementation Details

### Data Models

#### OHLCV Data Model (`data/ohlcv.py`)

- `OHLCVBar`: Represents a single OHLCV data bar with date, open, high, low, close, and volume
- `OHLCVData`: Collection of OHLCV bars with metadata and utility methods
- `FinancialData`: Utility class for loading and processing financial data

#### Fundamental Data Model (`data/fundamental_data.py`)

- `FinancialData`: Represents fundamental financial data including quarterly results, profit/loss statements, balance sheets, etc.
- Includes methods for accessing specific metrics and calculating financial ratios

### Agent Architecture

#### Base Agent (`agents/base_agent.py`)

- `BaseFinancialAgent`: Abstract base class for all financial agents
- Supports both Agno and LangChain frameworks with graceful fallback
- Provides tool management and agent state handling
- Allows continued use of LangChain agents without requiring conversion to Agno

#### Ben Graham Agent (`agents/ben_graham_agno.py`)

- Implements Benjamin Graham's value investing principles
- Analyzes earnings stability, financial strength, and valuation
- Generates investment signals with confidence scores and detailed reasoning

### Tools

#### Intraday Data Processor (`tools/intraday_data.py`)

- `IntradayDataProcessor`: Tools for processing and analyzing intraday financial data
- Includes methods for:
  - Data resampling
  - VWAP calculation
  - Gap detection
  - Technical indicator calculation
  - Support/resistance identification
  - Volume profile analysis

### LLM Integration

#### LLM Providers (`llm/providers.py`)

- `LLMManager`: Manager for multiple LLM providers
- Supports OpenAI, Anthropic, DeepSeek, Gemini, Groq, and LMStudio
- Provides unified interface for LLM calls with provider-specific implementations

### UI

#### Streamlit Application (`UI/app.py`)

- Main Streamlit interface with:
  - Ticker selection
  - LLM provider/model selection
  - Date range selection
  - Analysis results visualization
  - Technical indicator display
  - Interactive charts

## Current Implementation Status

The following components have been implemented:

1. **Data Models**: 
   - OHLCV data models with Pydantic
   - Fundamental data models with Pydantic

2. **Tools**: 
   - Intraday data processing tools
   - Technical indicator calculation

3. **Agent Architecture**: 
   - Base agent structure supporting both LangChain and Agno
   - No requirement to convert existing LangChain agents to Agno
   - Flexibility to use either framework based on specific needs

4. **Ben Graham Agent**: 
   - Agno-based implementation
   - Value investing analysis methods

5. **LLM Integration**: 
   - Multi-provider support
   - Unified interface for LLM calls

6. **Streamlit UI**: 
   - Main application structure
   - Analysis visualization
   - Technical indicator display

## Future Requirements

### Phase 1: Complete Agent Implementation

1. **Additional Agents**:
   - Complete the implementation of remaining investment strategy agents using LangChain (no need to convert to Agno)
   - Implement support agents (Portfolio Manager, Risk Manager, etc.)
   - Maintain compatibility with existing agent architecture

2. **Agent Orchestration**:
   - Develop a system for coordinating multiple agents
   - Implement weighted consensus mechanism for combining agent signals

### Phase 2: Enhanced Data Processing

1. **Real-time Data Integration**:
   - Connect to real-time market data APIs
   - Implement streaming data processing

2. **Alternative Data Sources**:
   - Integrate news sentiment analysis
   - Add social media monitoring
   - Include economic indicators

### Phase 3: Advanced Analysis

1. **Machine Learning Models**:
   - Implement predictive models for price movement
   - Develop anomaly detection for market events
   - Create portfolio optimization algorithms

2. **Backtesting Framework**:
   - Build comprehensive backtesting system
   - Implement performance metrics and reporting

### Phase 4: UI Enhancements

1. **Portfolio Management**:
   - Add portfolio tracking and management
   - Implement performance dashboards

2. **Alerts and Notifications**:
   - Create alert system for investment opportunities
   - Implement notification system for market events

3. **User Customization**:
   - Allow users to customize agent parameters
   - Enable saving and loading of analysis configurations

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys for LLM providers in environment variables

### Running the Application

```bash
streamlit run UI/app.py
```

## Contributing

Contributions to the Financial Agent System are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
