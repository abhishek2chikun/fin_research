#!/usr/bin/env python3
"""
Main Streamlit application for the Financial Agent System

This module provides the main Streamlit interface for the Financial Agent System,
serving as the home page and initializing the session state for the multi-page app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM providers
from llm.providers import llm_manager, LLMProvider

# Import Agno Teams integration
from bin.ui_integration import initialize_ui_integration

# Set page configuration
st.set_page_config(
    page_title="Financial Agent System",
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
    st.session_state.selected_master_agent = "stock_analysis_master"
    
if 'selected_slave_agents' not in st.session_state:
    st.session_state.selected_slave_agents = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = LLMProvider.LMSTUDIO.value

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "sufe-aiflm-lab_fin-r1"

if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=365), datetime.now())

if 'team_analysis_mode' not in st.session_state:
    st.session_state.team_analysis_mode = True




































        






# Initialize Agno integration with UI
initialize_ui_integration()


def main():
    """Main function for the home page"""
    # Display header
    st.title("💹 Financial Agent System")
    st.markdown("AI-powered investment analysis using master-slave agent architecture with Agno Teams")
    
    # Display welcome message and instructions
    st.header("Welcome to the Financial Agent System")
    
    st.markdown("""
    This system uses a master-slave agent architecture to analyze stocks and provide comprehensive investment insights.
    
    ### Features
    
    - **Master-Slave Architecture**: Hierarchical agent structure for coordinated analysis
    - **Stock Analysis**: Analyze individual stocks using multiple specialized agents
    - **Sectorial Analysis**: Find the best stocks in a sector with comparative rankings
    - **Portfolio Management**: Optimize and rebalance investment portfolios
    - **Multiple Investment Philosophies**: Value, Quality, Growth, Macro, and more
    - **Multiple LLM Providers**: Support for various LLM providers including LMStudio with financial models
    
    ### Getting Started
    
    1. Use the sidebar navigation to select the analysis type
    2. Choose a master agent (orchestrator) for your analysis
    3. Select slave agents (analysts) with different specializations
    4. Configure your analysis settings and run the analysis
    5. Review the comprehensive insights from multiple perspectives
    """)
    
    # Display available pages
    st.header("Available Analysis Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Analysis")
        st.markdown("Analyze individual stocks using AI agents with different investment strategies.")
        st.page_link("pages/1_stock_analysis.py", label="Go to Stock Analysis", icon="📈")
    
    with col2:
        st.subheader("Sectorial Analysis")
        st.markdown("Find the best stocks in a sector using AI agents with different investment strategies.")
        st.page_link("pages/2_sectorial_analysis.py", label="Go to Sectorial Analysis", icon="🏢")
    
    # Display system information
    st.header("System Information")
    
    # Display available LLM providers
    available_providers = llm_manager.get_available_providers()
    
    st.markdown(f"**Available LLM Providers:** {', '.join([provider.value for provider in available_providers])}")
    
    # Display available agents
    st.markdown("**Available Master Agents (Orchestrators):**")
    st.markdown("- Stock Analysis Master (Agno Teams)")
    st.markdown("- Sector Analysis Master (Agno Teams)")
    st.markdown("- Portfolio Master (Agno Teams)")

    st.markdown("**Available Slave Agents (Analysts):**")
    st.markdown("- Investment Philosophy: Ben Graham, Warren Buffett, Charlie Munger, Phil Fisher, etc.")
    st.markdown("- Analysis Domains: Technical, Valuation, Fundamental, Sentiment")
    st.markdown("- Support: Portfolio Manager, Risk Manager")
    
    # Display data sources
    st.markdown("**Data Sources:**")
    st.markdown("- Nifty Total Market List")


if __name__ == "__main__":
    main()
