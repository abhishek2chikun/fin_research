#!/usr/bin/env python3
"""
Stock Analysis page for the Financial Agent System

This module provides the stock analysis interface for analyzing individual stocks
using the master-slave agent architecture with Agno Teams.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data models
from data.ohlcv import OHLCVData, FinancialData

# Import LLM providers
from llm.providers import llm_manager, LLMProvider

# Import Agno Teams components
from bin.ui_integration import (
    get_all_available_agents,
    get_default_master_agent_for_page,
    display_master_agent_selection,
    display_slave_agent_selection,
    run_analysis_with_teams
)

# Import tools
from tools.intraday_data import IntradayDataProcessor

# Import shared utilities
from UI.utils import (
    load_available_tickers,
    display_ticker_selection,
    display_model_selection,
    display_date_range_selection
)

st.set_page_config(
    page_title="Stock Analysis - Financial Agent System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
    
if 'selected_agents' not in st.session_state:
    st.session_state.selected_agents = []

if 'selected_master_agent' not in st.session_state:
    st.session_state.selected_master_agent = None  # Will be set automatically based on page
    
if 'selected_slave_agents' not in st.session_state:
    st.session_state.selected_slave_agents = {}

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = LLMProvider.LMSTUDIO.value

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "sufe-aiflm-lab_fin-r1"

if 'team_analysis_mode' not in st.session_state:
    st.session_state.team_analysis_mode = True
    
if 'selected_agent_type' not in st.session_state:
    st.session_state.selected_agent_type = "investment_philosophy"

# Set the current page for master agent selection
CURRENT_PAGE = "stock_analysis"

def display_stock_analysis_results(agent_id, results):
    """Display results for individual stock analysis"""
    # Display results for each ticker
    for ticker, ticker_results in results.items():
        st.subheader(f"Analysis for {ticker}")
        
        # Display signal and confidence
        signal = ticker_results.get("signal", "neutral")
        confidence = ticker_results.get("confidence", 0.5)
        
        # Color-coded signal
        signal_color = "green" if signal == "bullish" else "red" if signal == "bearish" else "orange"
        st.markdown(f"<h3 style='color: {signal_color};'>Signal: {signal.upper()} (Confidence: {confidence:.1%})</h3>", unsafe_allow_html=True)
        
        # Display reasoning
        st.markdown("### Reasoning")
        st.markdown(ticker_results.get("reasoning", "No reasoning provided."))
        
        # Display metrics if available
        if "metrics" in ticker_results:
            st.markdown("### Metrics")
            metrics = ticker_results["metrics"]
            
            # Create metrics display
            cols = st.columns(min(len(metrics), 4))  # Max 4 columns
            for i, (metric, value) in enumerate(metrics.items()):
                col_idx = i % 4
                cols[col_idx].metric(
                    label=metric.replace("_", " ").title(),
                    value=f"{value:.2f}" if isinstance(value, (int, float)) else value
                )

# display_agent_type_selection function has been removed as requested


def display_team_analysis_results(results):
    """Display team analysis results with comprehensive view"""
    # Get the first ticker
    if not results or not isinstance(results, dict):
        st.error("No valid results to display")
        return
    
    # Display results for each ticker
    for ticker, ticker_results in results.items():
        st.subheader(f"Analysis for {ticker}")
        
        # Check if ticker_results is a dictionary
        if not isinstance(ticker_results, dict):
            st.warning(f"Invalid results format for {ticker}: {ticker_results}")
            continue
        
        # Create columns for overview and details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display signal and confidence
            signal = ticker_results.get("signal", "neutral")
            confidence = ticker_results.get("confidence", 0.5)
            
            # Color-coded signal
            signal_color = "green" if signal in ["buy", "strong buy", "bullish"] else "red" if signal in ["sell", "strong sell", "bearish"] else "orange"
            st.markdown(f"<h3 style='color: {signal_color};'>Signal: {signal.upper()} (Confidence: {confidence:.1%})</h3>", unsafe_allow_html=True)
        
        with col2:
            # Display strengths and weaknesses
            if "strengths" in ticker_results and isinstance(ticker_results["strengths"], list):
                st.markdown("#### Key Strengths")
                strengths = ticker_results["strengths"][:3] if len(ticker_results["strengths"]) > 0 else []  # Show top 3
                for strength in strengths:
                    st.markdown(f"✅ {strength}")
            
            if "weaknesses" in ticker_results and isinstance(ticker_results["weaknesses"], list):
                st.markdown("#### Key Concerns")
                weaknesses = ticker_results["weaknesses"][:3] if len(ticker_results["weaknesses"]) > 0 else []  # Show top 3
                for weakness in weaknesses:
                    st.markdown(f"⚠️ {weakness}")
        
        # Display reasoning
        st.markdown("### Analysis Summary")
        st.markdown(ticker_results.get("reasoning", "No reasoning provided."))
        
        # Display metrics if available
        if "metrics" in ticker_results and isinstance(ticker_results["metrics"], dict):
            st.markdown("### Key Metrics")
            metrics = ticker_results["metrics"]
            
            # Create a clean table of metrics
            metrics_data = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Format numeric values
                    if key.lower().endswith(('ratio', 'margin', 'growth', 'yield', 'return', 'turnover')):
                        formatted_value = f"{value:.2%}"
                    elif value > 1000000000:
                        formatted_value = f"${value/1000000000:.2f}B"
                    elif value > 1000000:
                        formatted_value = f"${value/1000000:.2f}M"
                    elif value > 1000:
                        formatted_value = f"${value/1000:.2f}K"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                metrics_data.append({"Metric": key, "Value": formatted_value})
            
            if metrics_data:
                st.table(pd.DataFrame(metrics_data))
        
        # Display analysis by type
        if "analysis_by_type" in ticker_results and isinstance(ticker_results["analysis_by_type"], dict):
            st.markdown("### Analysis by Investment Philosophy")
            
            try:
                # Safely get keys for tabs
                analysis_types = list(ticker_results["analysis_by_type"].keys())
                
                if analysis_types:  # Only create tabs if there are analysis types
                    analysis_tabs = st.tabs(analysis_types)
                    
                    for i, (analysis_type, analysis) in enumerate(ticker_results["analysis_by_type"].items()):
                        if i < len(analysis_tabs):  # Make sure we don't exceed the number of tabs
                            with analysis_tabs[i]:
                                # Check if analysis is a dictionary
                                if not isinstance(analysis, dict):
                                    st.warning(f"Invalid analysis format for {analysis_type}: {analysis}")
                                    continue
                                
                                # Display signal and confidence for this analysis type
                                signal = analysis.get("signal", "neutral")
                                confidence = analysis.get("confidence", 0.5)
                                
                                # Color-coded signal
                                signal_color = "green" if signal in ["buy", "strong buy", "bullish"] else "red" if signal in ["sell", "strong sell", "bearish"] else "orange"
                                st.markdown(f"<h4 style='color: {signal_color};'>Signal: {signal.upper()} (Confidence: {confidence:.1%})</h4>", unsafe_allow_html=True)
                                
                                # Display reasoning
                                st.markdown("#### Reasoning")
                                st.markdown(analysis.get("reasoning", "No reasoning provided."))
                else:
                    st.info("No investment philosophy analysis available")
            except Exception as e:
                st.error(f"Error displaying analysis by type: {str(e)}")


def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.analysis_results:
        return
    
    st.header("Stock Analysis Results")
    
    # Check if team analysis mode
    if st.session_state.team_analysis_mode:
        # For team analysis, display comprehensive results
        display_team_analysis_results(st.session_state.analysis_results)
    else:
        # Legacy display for individual agent results
        agent_ids = list(st.session_state.analysis_results.keys())
        if not agent_ids:
            st.info("No analysis results available.")
            return
        
        # Get available agents for naming
        available_agents = get_all_available_agents()
        master_agents = available_agents.get("master", [])
        agent_names = {agent["id"]: agent["name"] for agent in master_agents}
        
        tabs = st.tabs([agent_names.get(agent_id, agent_id) for agent_id in agent_ids])
        
        # Display results for each agent
        for i, agent_id in enumerate(agent_ids):
            with tabs[i]:
                agent_results = st.session_state.analysis_results[agent_id]
                display_stock_analysis_results(agent_id, agent_results)

def run_team_analysis_button():
    """Display run analysis button for team-based analysis"""
    if not st.session_state.selected_tickers:
        st.warning("Please select at least one ticker first.")
        return
    
    if not st.session_state.selected_master_agent:
        st.warning("Please select a master agent.")
        return
    
    if not st.session_state.selected_slave_agents:
        st.warning("Please select at least one slave agent.")
        return
    
    if st.button("Run Analysis with Teams", type="primary"):
        with st.spinner("Running analysis... This may take a few minutes."):
            try:
                # Use current date for end_date
                end_date = datetime.now()
                # Use 1 year ago for start_date (for any data that might need historical context)
                start_date = end_date - timedelta(days=365)
                
                # Run analysis using Agno Teams
                results = run_analysis_with_teams(
                    tickers=st.session_state.selected_tickers,
                    master_agent_id=st.session_state.selected_master_agent,
                    slave_agent_ids=st.session_state.selected_slave_agents,
                    provider=st.session_state.llm_provider,
                    model=st.session_state.llm_model,
                    start_date=start_date,
                    end_date=end_date,
                    is_sector_analysis=False,
                    is_portfolio_analysis=False
                )
                
                # Store results in session state
                st.session_state.analysis_results = results
                
                # Log completion
                logger.info(f"Analysis completed for {st.session_state.selected_tickers}")
                
                # Force rerun to display results
                st.rerun()
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                logger.error(f"Error in analysis: {str(e)}")


def main():
    """Main function for the stock analysis page"""
    st.title("📈 Stock Analysis")
    st.markdown("Analyze individual stocks using AI agents with different investment philosophies")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Analysis Settings")
        
        # Display ticker selection
        display_ticker_selection()
        
        # Set the master agent for this page
        if not st.session_state.selected_master_agent:
            st.session_state.selected_master_agent = get_default_master_agent_for_page(CURRENT_PAGE)
        
        # Agent type selection has been removed as requested
        
        # Select slave agents (without using agent_type parameter directly)
        selected_slaves = display_slave_agent_selection(
            st.session_state.selected_master_agent, 
            multiselect=True
        )
        # Don't override the entire selected_slave_agents dictionary, just keep the value for this master agent
        # The display_slave_agent_selection function already updates st.session_state.selected_slave_agents[master_agent_id]
        
        # Display model selection
        display_model_selection()
        
        # Display run analysis button
        run_team_analysis_button()
    
    # Display analysis results
    display_analysis_results()

if __name__ == "__main__":
    main()
