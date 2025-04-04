# Financial Agent System

## Overview

AI-driven investment analysis platform using master-slave agent architecture built on the Agno framework with coordinate mode. The system performs stock analysis, sector comparisons, and portfolio optimization using various investment philosophies implemented as specialized agents.

## Directory Structure

```
fin_agent/
├── master_agent/             # Orchestrator agents and coordination
│   ├── stock_analysis_master.py  # Stock analysis coordinator
│   ├── sector_analysis_master.py # Sector analysis coordinator
│   ├── portfolio_master.py    # Portfolio optimization coordinator
│   ├── team_coordinator.py    # Team coordination utilities
│   ├── base_agent.py          # Base agent classes
│   ├── agno_wrapper.py        # Wrapper for Agno components
│   ├── progress_util.py       # Progress tracking utilities
│   └── ui_integration.py      # UI integration for Agno Teams
├── slave_agent/              # Specialized analysis agents
│   ├── ben_graham_agno.py    # Value investing (Graham)
│   ├── warren_buffett_agno.py # Value + Quality (Buffett)
│   ├── phil_fisher_agno.py   # Growth investing (Fisher)
│   ├── charlie_munger_agno.py # Quality investing (Munger)
│   ├── bill_ackman_agno.py   # Activist investing (Ackman)
│   ├── cathie_wood_agno.py   # Innovation investing (Wood)
│   ├── stanley_druckenmiller_agno.py # Macro investing (Druckenmiller)
│   ├── fundamentals_agno.py  # Fundamental analysis
│   ├── technicals_agno.py    # Technical analysis
│   ├── sentiment_agno.py     # Sentiment analysis
│   ├── risk_manager_agno.py  # Risk assessment
│   └── portfolio_manager_agno.py # Portfolio management
├── data/                     # Data models
│   ├── fundamental_data.py   # Financial metrics and ratios
│   └── ohlcv.py              # Price and volume data
├── UI/                       # Streamlit interface
│   ├── app.py                # Main application
│   ├── utils.py              # UI utilities
│   └── pages/                # Analysis pages
│       ├── 1_stock_analysis.py    # Individual stock analysis
│       ├── 2_sector_analysis.py   # Sector comparison
│       └── 3_portfolio_management.py # Portfolio tools
└── utils/                    # Shared utilities
    └── financial_utils.py    # Financial calculations
```

## Technical Architecture

### Master-Slave Agent Architecture with Coordinate Mode

1. **Master Agents (Coordinators)**
   - Coordinate and manage analysis workflows using Agno's coordinate mode
   - Delegate tasks to appropriate slave agents
   - Synthesize results into a comprehensive analysis
   - Three main coordinators:
     - Stock Analysis Master: Individual stock analysis
     - Sector Analysis Master: Sector-wide comparative analysis
     - Portfolio Master: Portfolio optimization and risk assessment

2. **Slave Agents (Specialists)**
   - Investment philosophy-based agents (Graham, Buffett, Fisher, etc.)
   - Technical specialists (fundamentals, technicals, sentiment)
   - Support functions (risk management, portfolio optimization)
   - Each agent implements a specific analytical approach

3. **Agno Teams Integration with Coordinate Mode**
   - Team Leader (Master Agent) delegates tasks to team members (Slave Agents)
   - Team Leader synthesizes outputs into a cohesive response
   - Progress tracking and error handling
   - Streamlined communication between agents

4. **UI Integration**
   - Streamlit-based interface with multiple analysis pages
   - Master and slave agent selection
   - Analysis parameter configuration
   - Comprehensive visualization of results

### LLM Integration

- **Multiple Providers Support**
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude models)
  - LM Studio (local models)
  - Configurable via the UI

- **Financial Domain Optimization**
  - Specialized prompts for financial analysis
  - Investment philosophy-specific reasoning
  - Structured output formats for consistent analysis

## Development Guidelines

1. **Code Style**
   - Follow PEP 8 standards
   - Use type hints consistently
   - Document with comprehensive docstrings
   - Maintain consistent import structure (from master_agent instead of agno_agent)

2. **Testing**
   - Unit tests for data models and utilities
   - Integration tests for agent interactions
   - Test new features before merging
   - Verify coordinate mode works correctly

3. **Error Handling**
   - Graceful degradation on API failures
   - Comprehensive logging
   - User-friendly error messages in UI
   - Handle team coordination failures

4. **Performance**
   - Profile and optimize slow operations
   - Cache expensive calculations
   - Use async where appropriate
   - Optimize team communication
