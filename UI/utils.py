#!/usr/bin/env python3
"""
Utility functions for the Financial Agent System UI

This module provides shared utility functions used by multiple pages
in the Streamlit UI for the Financial Agent System.
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import types for type checking only
from typing import TYPE_CHECKING, Type, Any
if TYPE_CHECKING:
    from agents import BaseFinancialAgent
    # Only import specific agent classes when needed for type checking

# Import LLM providers
from llm.providers import llm_manager, LLMProvider


def load_available_tickers() -> List[str]:
    """Load list of available tickers from the Nifty Total Market list"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Historical_data')
    nifty_list_path = os.path.join(data_dir, 'ind_niftytotalmarket_list.csv')
    
    if not os.path.exists(nifty_list_path):
        return []
    
    # Read the CSV file with company information
    try:
        df = pd.read_csv(nifty_list_path)
        # Return the list of symbols
        return sorted(df['Symbol'].tolist())
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return []


def load_sectors() -> Dict[str, List[str]]:
    """Load sectors and their stocks from the Nifty Total Market list"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Historical_data')
    nifty_list_path = os.path.join(data_dir, 'ind_niftytotalmarket_list.csv')
    
    if not os.path.exists(nifty_list_path):
        return {}
    
    # Read the CSV file with company information
    try:
        df = pd.read_csv(nifty_list_path)
        
        # Group by industry and get symbols for each industry
        sectors = {}
        for industry, group in df.groupby('Industry'):
            sectors[industry] = sorted(group['Symbol'].tolist())
        
        return sectors
    except Exception as e:
        st.error(f"Error loading sectors: {e}")
        return {}


def load_available_sectors() -> List[str]:
    """Load list of available sectors from the Nifty Total Market list"""
    sectors = load_sectors()
    return sorted(list(sectors.keys()))


def dynamically_import_agent_class(agent_module_path: str, agent_class_name: str) -> Type:
    """
    Dynamically import a class from a module path
    
    Args:
        agent_module_path: Path to the module (e.g., 'master_agent.stock_analysis_master')
        agent_class_name: Name of the class to import (e.g., 'StockAnalysisMaster')
        
    Returns:
        The imported class
    """
    from importlib import import_module
    
    try:
        module = import_module(agent_module_path)
        agent_class = getattr(module, agent_class_name)
        return agent_class
    except (ImportError, AttributeError) as e:
        st.error(f"Error importing agent class {agent_class_name} from {agent_module_path}: {e}")
        return None


def get_available_agents() -> List[Dict[str, Any]]:
    """Get list of available agents"""
    # Import agent modules dynamically to avoid circular imports
    from importlib import import_module
    
    # List of available agents
    available_agents = []
    
    # Add Agno agents first
    available_agents.append({"name": "Stock Analysis Agent", "id": "stock_analysis", "class": None})
    available_agents.append({"name": "Sector Analysis Agent", "id": "sector_analysis", "class": None})
    
    # Add LangChain agents
    try:
        # Try to import each agent, but continue if one fails
        try:
            # Ben Graham agent
            from agents.ben_graham import ben_graham_agent
            available_agents.append({"name": "Ben Graham Value Investor", "id": "ben_graham", "class": ben_graham_agent})
        except ImportError:
            pass
        
        try:
            # Bill Ackman agent
            from agents.bill_ackman import bill_ackman_agent
            available_agents.append({"name": "Bill Ackman Investor", "id": "bill_ackman", "class": bill_ackman_agent})
        except ImportError:
            pass
        
        try:
            # Warren Buffett agent
            from agents.warren_buffett import warren_buffett_agent
            available_agents.append({"name": "Warren Buffett Investor", "id": "warren_buffett", "class": warren_buffett_agent})
        except ImportError:
            pass
        
        try:
            # Fundamentals agent
            from agents.fundamentals import fundamentals_agent
            available_agents.append({"name": "Fundamentals Analyst", "id": "fundamentals", "class": fundamentals_agent})
        except ImportError:
            pass
        
        try:
            # Technicals agent
            from agents.technicals import technical_analyst_agent
            available_agents.append({"name": "Technical Analyst", "id": "technicals", "class": technical_analyst_agent})
        except ImportError:
            pass
        
        try:
            # Sentiment agent
            from agents.sentiment import sentiment_agent
            available_agents.append({"name": "Sentiment Analyst", "id": "sentiment", "class": sentiment_agent})
        except ImportError:
            pass
        
        # Always add portfolio and risk management agents
        try:
            from agents.portfolio_manager import portfolio_management_agent
            available_agents.append({"name": "Portfolio Manager", "id": "portfolio_management", "class": portfolio_management_agent})
        except ImportError:
            pass
        
        try:
            from agents.risk_manager import risk_management_agent
            available_agents.append({"name": "Risk Manager", "id": "risk_management", "class": risk_management_agent})
        except ImportError:
            pass
            
    except Exception as e:
        st.error(f"Error loading agents: {e}")
    
    return available_agents


def run_analysis(tickers: List[str], agents: List[str], provider: str, model: str, 
                start_date: datetime, end_date: datetime, 
                is_sector_analysis: bool = False) -> Dict[str, Any]:
    """Run financial analysis using the selected agents"""
    # Create agent state
    state = {
        "data": {
            "tickers": tickers,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "analyst_signals": {},
            "is_sector_analysis": is_sector_analysis
        },
        "metadata": {
            "model_name": model,
            "model_provider": provider,
            "show_reasoning": True
        },
        "messages": []
    }
    
    # Get available agents (this function handles dynamic imports)
    available_agents = get_available_agents()
    agent_map = {agent["id"]: agent["class"] for agent in available_agents}
    
    results = {}
    
    # Check if we're using Agno agents
    if any(agent_id in ["stock_analysis", "sector_analysis"] for agent_id in agents):
        # Use Agno workflow
        from bin.workflow import run_financial_analysis
        
        # Set selected_sector if this is sector analysis
        selected_sector = st.session_state.selected_sector if is_sector_analysis else None
        
        # Run the financial analysis workflow
        workflow_results = run_financial_analysis(
            tickers=tickers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            show_reasoning=True,
            selected_agents=agents,
            model_name=model,
            model_provider=provider,
            is_sector_analysis=is_sector_analysis,
            selected_sector=selected_sector
        )
        
        # Extract results
        results = workflow_results["analyst_signals"]
    else:
        # Use traditional LangChain approach
        for agent_id in agents:
            if agent_id in agent_map:
                agent_func = agent_map[agent_id]
                
                # Run analysis
                agent_result = agent_func(state)
                
                # Update state for next agent
                state = agent_result
                
                # Extract results
                if agent_id in agent_result["data"]["analyst_signals"]:
                    results[agent_id] = agent_result["data"]["analyst_signals"][agent_id]
    
    return results


def display_ticker_selection():
    """Display ticker selection UI"""
    st.sidebar.header("Stock Selection")
    
    # Load available tickers
    available_tickers = load_available_tickers()
    
    if not available_tickers:
        st.sidebar.warning("No ticker data found in Historical_data directory.")
        return
    
    # Multi-select for tickers
    selected_tickers = st.sidebar.multiselect(
        "Select stocks to analyze:",
        options=available_tickers,
        default=st.session_state.selected_tickers if st.session_state.selected_tickers else None
    )
    
    # Update session state
    st.session_state.selected_tickers = selected_tickers


def display_agent_selection():
    """Display agent selection UI"""
    st.sidebar.header("Agent Selection")
    
    # Get available agents
    available_agents = get_available_agents()
    available_agent_ids = [agent["id"] for agent in available_agents]
    
    # Initialize session state if needed
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = []
    
    # Set default agents
    if not st.session_state.selected_agents:
        # Default to stock_analysis if available, otherwise use the first available agent
        if "stock_analysis" in available_agent_ids:
            st.session_state.selected_agents = ["stock_analysis"]
        elif available_agent_ids:
            st.session_state.selected_agents = [available_agent_ids[0]]
        else:
            st.session_state.selected_agents = []
    
    # Filter out any agents that are no longer available
    valid_agents = [agent_id for agent_id in st.session_state.selected_agents 
                    if agent_id in available_agent_ids]
    
    # If all selected agents are invalid, reset to default
    if not valid_agents and available_agent_ids:
        if "stock_analysis" in available_agent_ids:
            valid_agents = ["stock_analysis"]
        else:
            valid_agents = [available_agent_ids[0]]
    
    # Update session state with valid agents
    st.session_state.selected_agents = valid_agents
    
    # Multi-select for agents
    selected_agents = st.sidebar.multiselect(
        "Select agents to use:",
        options=available_agent_ids,
        format_func=lambda x: next((agent["name"] for agent in available_agents if agent["id"] == x), x),
        default=valid_agents
    )
    
    # Update session state
    st.session_state.selected_agents = selected_agents


def display_sector_selection():
    """Display sector selection UI"""
    st.sidebar.header("Sector Selection")
    
    # Load available sectors
    sectors = load_sectors()
    
    if not sectors:
        st.sidebar.warning("No sector data found.")
        return
    
    # Select sector
    selected_sector = st.sidebar.selectbox(
        "Select sector to analyze:",
        options=list(sectors.keys()),
        index=list(sectors.keys()).index(st.session_state.selected_sector) if st.session_state.selected_sector in sectors else 0
    )
    
    # Update session state
    st.session_state.selected_sector = selected_sector
    
    # Display stocks in the selected sector
    if selected_sector in sectors:
        st.sidebar.markdown(f"**Stocks in {selected_sector}:** {len(sectors[selected_sector])}")
        with st.sidebar.expander("View stocks"):
            st.write(", ".join(sectors[selected_sector]))


def display_model_selection():
    """Display LLM model selection UI"""
    st.sidebar.header("Model Selection")
    
    # Get available providers
    available_providers = [provider.value for provider in llm_manager.get_available_providers()]
    
    if not available_providers:
        st.sidebar.warning("No LLM providers available. Please check configuration.")
        return
        
    # Get initialized providers (those with API keys)
    from llm.providers import INITIALIZED_CLIENTS
    initialized_providers = [p for p in available_providers if INITIALIZED_CLIENTS.get(p, False)]
    
    # Select provider with status indicator
    provider_options = available_providers
    provider_display = {}
    
    for p in provider_options:
        if p in initialized_providers:
            provider_display[p] = f"{p} ✓"
        else:
            provider_display[p] = f"{p} (API key not set)"
    
    # Find default index
    default_index = 0
    if st.session_state.llm_provider in available_providers:
        default_index = available_providers.index(st.session_state.llm_provider)
    
    # Select provider
    provider = st.sidebar.selectbox(
        "Select LLM provider:",
        options=provider_options,
        format_func=lambda x: provider_display.get(x, x),
        index=default_index
    )
    
    # Show warning if provider not initialized
    if provider not in initialized_providers:
        st.sidebar.warning(f"⚠️ {provider} API key not configured. Set the {provider.upper()}_API_KEY environment variable.")
    
    # Get available models for the selected provider
    provider_enum = LLMProvider(provider)
    
    # First try to get models from the available_models dictionary
    available_models = llm_manager.available_models.get(provider_enum, [])
    
    # If no models found in dictionary, try getting them directly
    if not available_models:
        available_models = llm_manager.get_available_models(provider_enum)
    
    # Ensure we have at least one model
    if not available_models:
        available_models = [llm_manager.get_default_model(provider_enum)]
    
    # Format model names for display
    model_options = available_models
    model_display = {model: llm_manager.get_model_display_name(model) for model in model_options}
    
    # Determine default index
    default_index = 0
    if st.session_state.llm_model in model_options:
        default_index = model_options.index(st.session_state.llm_model)
    
    # Select model
    model = st.sidebar.selectbox(
        "Select model:",
        options=model_options,
        format_func=lambda x: model_display.get(x, x),
        index=default_index
    )
    
    # Update session state
    st.session_state.llm_provider = provider
    st.session_state.llm_model = model


def display_date_range_selection():
    """Display date range selection UI"""
    st.sidebar.header("Date Range")
    
    # Date range selection
    start_date = st.sidebar.date_input(
        "Start date:",
        value=st.session_state.date_range[0]
    )
    
    end_date = st.sidebar.date_input(
        "End date:",
        value=st.session_state.date_range[1]
    )
    
    # Update session state
    st.session_state.date_range = (start_date, end_date)


def run_analysis_button(is_sector_analysis=False):
    """Display run analysis button"""
    st.sidebar.header("Run Analysis")
    
    # Check if required selections are made
    if is_sector_analysis:
        if not st.session_state.selected_sector:
            st.sidebar.warning("Please select a sector.")
            return
    else:
        if not st.session_state.selected_tickers:
            st.sidebar.warning("Please select at least one ticker.")
            return
    
    if not st.session_state.selected_agents:
        st.sidebar.warning("Please select at least one agent.")
        return
    
    # Run analysis button
    button_text = "Analyze Sector" if is_sector_analysis else "Analyze Stocks"
    if st.sidebar.button(button_text, type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            # Get selected parameters
            provider = st.session_state.llm_provider
            model = st.session_state.llm_model
            agents = st.session_state.selected_agents
            start_date, end_date = st.session_state.date_range
            
            if is_sector_analysis:
                # Get tickers for the selected sector
                sectors = load_sectors()
                tickers = sectors.get(st.session_state.selected_sector, [])
                if not tickers:
                    st.error(f"No stocks found for sector: {st.session_state.selected_sector}")
                    return
            else:
                tickers = st.session_state.selected_tickers
            
            # Run analysis
            results = run_analysis(tickers, agents, provider, model, start_date, end_date, is_sector_analysis)
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            st.success(f"Analysis completed for {len(tickers)} tickers using {len(agents)} agents.")
