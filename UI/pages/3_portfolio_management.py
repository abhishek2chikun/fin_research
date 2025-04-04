"""
Portfolio Management Page for the Financial Agent UI.
This page provides portfolio optimization and management using the master-slave agent architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import json

# Import utilities and data
from UI.utils import (
    load_available_tickers,
    load_available_sectors
)

# Import Agno Teams integration
from bin.ui_integration import (
    get_available_master_agents,
    get_available_slave_agents,
    get_all_available_agents,
    get_default_master_agent_for_page,
    display_master_agent_selection,
    display_slave_agent_selection,
    run_analysis_with_teams
)

# Import LLM Providers
from bin.llm_providers import LLMProvider

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Portfolio Management",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

if 'selected_master_agent' not in st.session_state:
    st.session_state.selected_master_agent = None  # Will be set automatically based on page

if 'selected_slave_agents' not in st.session_state:
    st.session_state.selected_slave_agents = []

if 'portfolio_analysis_results' not in st.session_state:
    st.session_state.portfolio_analysis_results = {}

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = LLMProvider.LMSTUDIO.value

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "sufe-aiflm-lab_fin-r1"

if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=365), datetime.now())

if 'team_analysis_mode' not in st.session_state:
    st.session_state.team_analysis_mode = True
    
# Set the current page for master agent selection
CURRENT_PAGE = "portfolio_management"


def display_portfolio_input():
    """Allow user to input or upload portfolio data"""
    input_method = st.radio(
        "Portfolio Input Method",
        options=["Manual Entry", "Upload CSV", "Use Sample Portfolio"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        display_manual_portfolio_entry()
    elif input_method == "Upload CSV":
        display_csv_upload()
    else:
        load_sample_portfolio()


def display_manual_portfolio_entry():
    """Display form for manual portfolio entry"""
    with st.form("portfolio_form"):
        # Get available tickers
        all_tickers = load_available_tickers()
        
        # Create a container for added stocks
        if 'portfolio_entries' not in st.session_state:
            st.session_state.portfolio_entries = [{"ticker": "", "weight": 0.0, "cost_basis": 0.0}]
        
        # Display each entry in the portfolio
        for i, entry in enumerate(st.session_state.portfolio_entries):
            cols = st.columns([3, 2, 2, 1])
            with cols[0]:
                ticker = st.selectbox(
                    f"Stock {i+1}",
                    options=[""] + all_tickers,
                    key=f"ticker_{i}"
                )
                st.session_state.portfolio_entries[i]["ticker"] = ticker
            
            with cols[1]:
                weight = st.number_input(
                    "Weight (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(entry["weight"] * 100) if entry["weight"] else 0.0,
                    key=f"weight_{i}"
                )
                st.session_state.portfolio_entries[i]["weight"] = weight / 100
            
            with cols[2]:
                cost_basis = st.number_input(
                    "Cost Basis ($)",
                    min_value=0.0,
                    value=float(entry["cost_basis"]) if entry["cost_basis"] else 0.0,
                    key=f"cost_{i}"
                )
                st.session_state.portfolio_entries[i]["cost_basis"] = cost_basis
            
            with cols[3]:
                if i > 0 and st.button("❌", key=f"remove_{i}"):
                    st.session_state.portfolio_entries.pop(i)
                    st.rerun()
        
        # Add more button
        if st.button("Add Stock"):
            st.session_state.portfolio_entries.append({"ticker": "", "weight": 0.0, "cost_basis": 0.0})
            st.rerun()
        
        # Submit button
        submit = st.form_submit_button("Save Portfolio")
        
        if submit:
            # Validate that weights sum to 100%
            total_weight = sum(entry["weight"] for entry in st.session_state.portfolio_entries if entry["ticker"])
            if abs(total_weight - 1.0) > 0.01:
                st.error(f"Weights must sum to 100%. Current total: {total_weight*100:.1f}%")
                return
            
            # Save to portfolio
            portfolio = {}
            for entry in st.session_state.portfolio_entries:
                if entry["ticker"]:
                    portfolio[entry["ticker"]] = {
                        "weight": entry["weight"],
                        "cost_basis": entry["cost_basis"]
                    }
            
            st.session_state.portfolio = portfolio
            st.success("Portfolio saved successfully!")


def display_csv_upload():
    """Allow user to upload portfolio as CSV"""
    st.markdown("Upload a CSV file with your portfolio. Format: ticker,weight,cost_basis")
    st.markdown("Example: AAPL,0.25,150.00")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            content = StringIO(uploaded_file.getvalue().decode('utf-8'))
            portfolio_df = pd.read_csv(content, header=None)
            
            if portfolio_df.shape[1] < 2:
                st.error("CSV must have at least two columns: ticker and weight.")
                return
            
            # Map columns
            portfolio_df.columns = (['ticker', 'weight'] + 
                                   (['cost_basis'] if portfolio_df.shape[1] >= 3 else []) + 
                                   [f'col_{i}' for i in range(3, portfolio_df.shape[1])])
            
            # Create portfolio dict
            portfolio = {}
            for _, row in portfolio_df.iterrows():
                portfolio[row['ticker']] = {
                    "weight": float(row['weight']),
                    "cost_basis": float(row['cost_basis']) if 'cost_basis' in row else 0.0
                }
            
            # Normalize weights if needed
            total_weight = sum(entry["weight"] for entry in portfolio.values())
            if abs(total_weight - 1.0) > 0.01:
                for ticker in portfolio:
                    portfolio[ticker]["weight"] /= total_weight
                st.warning(f"Weights have been normalized to sum to 100%. Original sum: {total_weight*100:.1f}%")
            
            # Save to session state
            st.session_state.portfolio = portfolio
            st.success(f"Portfolio with {len(portfolio)} stocks loaded successfully!")
            
            # Update portfolio entries for manual editing
            st.session_state.portfolio_entries = [
                {"ticker": ticker, "weight": data["weight"], "cost_basis": data["cost_basis"]}
                for ticker, data in portfolio.items()
            ]
            
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")


def load_sample_portfolio():
    """Load a sample portfolio for demonstration"""
    sample_portfolio = {
        "AAPL": {"weight": 0.20, "cost_basis": 150.00},
        "MSFT": {"weight": 0.15, "cost_basis": 280.00},
        "AMZN": {"weight": 0.15, "cost_basis": 120.00},
        "GOOGL": {"weight": 0.10, "cost_basis": 135.00},
        "TSLA": {"weight": 0.10, "cost_basis": 230.00},
        "NVDA": {"weight": 0.15, "cost_basis": 400.00},
        "JPM": {"weight": 0.10, "cost_basis": 145.00},
        "V": {"weight": 0.05, "cost_basis": 240.00}
    }
    
    st.session_state.portfolio = sample_portfolio
    st.session_state.portfolio_entries = [
        {"ticker": ticker, "weight": data["weight"], "cost_basis": data["cost_basis"]}
        for ticker, data in sample_portfolio.items()
    ]
    
    st.success("Sample portfolio loaded successfully!")


def display_model_selection():
    """Display LLM provider and model selection"""
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox(
            "LLM Provider",
            options=[p.value for p in LLMProvider],
            index=[p.value for p in LLMProvider].index(st.session_state.llm_provider) if st.session_state.llm_provider in [p.value for p in LLMProvider] else 0
        )
        st.session_state.llm_provider = provider
    
    with col2:
        # Different models based on provider
        if provider == LLMProvider.OPENAI.value:
            models = ["gpt-4", "gpt-3.5-turbo"]
        elif provider == LLMProvider.ANTHROPIC.value:
            models = ["claude-2", "claude-instant-1"]
        elif provider == LLMProvider.LMSTUDIO.value:
            models = ["sufe-aiflm-lab_fin-r1", "mistralai_mistral-7b-instruct", "meta-llama_llama-2-7b-chat"]
        else:
            models = ["default"]
        
        model = st.selectbox(
            "Model",
            options=models,
            index=models.index(st.session_state.llm_model) if st.session_state.llm_model in models else 0
        )
        st.session_state.llm_model = model


def display_date_range_selection():
    """Display date range selection for the analysis period"""
    start_date, end_date = st.date_input(
        "Select analysis time period",
        value=st.session_state.date_range,
        min_value=datetime(2010, 1, 1),
        max_value=datetime.now()
    )
    
    if start_date and end_date:
        st.session_state.date_range = (start_date, end_date)


def run_portfolio_analysis_button():
    """Display run analysis button for team-based portfolio analysis"""
    if not st.session_state.portfolio:
        st.warning("Please create or upload a portfolio first.")
        return
    
    if not st.session_state.selected_master_agent:
        st.warning("Please select a master agent.")
        return
    
    if not st.session_state.selected_slave_agents:
        st.warning("Please select at least one slave agent.")
        return
    
    analysis_type = st.radio(
        "Analysis Type",
        options=["Portfolio Review", "Rebalancing Recommendations", "Risk Assessment"],
        horizontal=True
    )
    
    if st.button("Run Portfolio Analysis"):
        with st.spinner(f"Running {analysis_type.lower()}... This may take several minutes."):
            try:
                # Extract tickers from portfolio
                tickers = list(st.session_state.portfolio.keys())
                
                # Convert portfolio to format expected by the analysis
                portfolio_data = {
                    "tickers": tickers,
                    "weights": [st.session_state.portfolio[ticker]["weight"] for ticker in tickers],
                    "cost_basis": [st.session_state.portfolio[ticker]["cost_basis"] for ticker in tickers],
                    "analysis_type": analysis_type.lower().replace(" ", "_")
                }
                
                # Run analysis using Agno Teams
                results = run_analysis_with_teams(
                    tickers=tickers,
                    master_agent_id=st.session_state.selected_master_agent,
                    slave_agent_ids=st.session_state.selected_slave_agents,
                    provider=st.session_state.llm_provider,
                    model=st.session_state.llm_model,
                    start_date=st.session_state.date_range[0],
                    end_date=st.session_state.date_range[1],
                    is_sector_analysis=False,
                    is_portfolio_analysis=True,
                    portfolio_data=portfolio_data
                )
                
                # Store results in session state
                st.session_state.portfolio_analysis_results = results
                st.session_state.analysis_type = analysis_type
                
                # Log completion
                logger.info(f"Portfolio analysis completed: {analysis_type}")
                
                # Force rerun to display results
                st.rerun()
            except Exception as e:
                st.error(f"Error in portfolio analysis: {str(e)}")
                logger.error(f"Error in portfolio analysis: {str(e)}")


def display_portfolio_allocation_chart():
    """Display portfolio allocation chart"""
    if not st.session_state.portfolio:
        return
    
    # Create dataframe for visualization
    portfolio_df = pd.DataFrame([
        {"Ticker": ticker, "Weight": data["weight"] * 100, "Cost Basis": data["cost_basis"]}
        for ticker, data in st.session_state.portfolio.items()
    ])
    
    # Create pie chart
    fig = px.pie(
        portfolio_df, 
        values="Weight", 
        names="Ticker",
        title="Current Portfolio Allocation",
        hover_data=["Cost Basis"],
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)


def display_portfolio_summary():
    """Display current portfolio summary"""
    if not st.session_state.portfolio:
        return
    
    st.subheader("Current Portfolio")
    
    # Display allocation chart
    display_portfolio_allocation_chart()
    
    # Display portfolio details as a table
    portfolio_details = []
    for ticker, data in st.session_state.portfolio.items():
        portfolio_details.append({
            "Ticker": ticker,
            "Weight": f"{data['weight']*100:.1f}%",
            "Cost Basis": f"${data['cost_basis']:.2f}"
        })
    
    # Sort by weight descending
    portfolio_details.sort(key=lambda x: float(x["Weight"].strip('%')), reverse=True)
    
    # Display as a table
    st.table(pd.DataFrame(portfolio_details))


def display_rebalancing_recommendations(results):
    """Display rebalancing recommendations from analysis"""
    if "portfolio_recommendations" not in results:
        st.warning("No rebalancing recommendations available.")
        return
    
    recommendations = results["portfolio_recommendations"]
    
    # Extract current and target allocations
    current_allocations = []
    target_allocations = []
    
    for ticker, data in st.session_state.portfolio.items():
        current_allocations.append({
            "Ticker": ticker,
            "Allocation": data["weight"],
            "Type": "Current"
        })
    
    if "target_weights" in recommendations:
        for ticker, weight in recommendations["target_weights"].items():
            target_allocations.append({
                "Ticker": ticker,
                "Allocation": weight,
                "Type": "Target"
            })
    
    # Combine and create dataframe
    all_allocations = pd.DataFrame(current_allocations + target_allocations)
    
    if not all_allocations.empty and "Target" in all_allocations["Type"].values:
        # Create grouped bar chart comparing current vs target
        fig = px.bar(
            all_allocations,
            x="Ticker",
            y="Allocation",
            color="Type",
            barmode="group",
            title="Current vs. Recommended Portfolio Allocation",
            color_discrete_map={"Current": "lightblue", "Target": "darkblue"},
            text_auto=".0%"
        )
        
        fig.update_layout(yaxis_tickformat=".0%")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display trade recommendations
    if "trades" in recommendations:
        st.subheader("Recommended Trades")
        
        trades = recommendations["trades"]
        trade_data = []
        
        for ticker, action in trades.items():
            if isinstance(action, dict):
                direction = action.get("action", "")
                amount = action.get("amount", 0)
                
                # Skip entries with no amount
                if amount == 0:
                    continue
                
                trade_data.append({
                    "Ticker": ticker,
                    "Action": direction.capitalize(),
                    "Change": f"{amount*100:+.1f}%" if isinstance(amount, float) else amount
                })
            elif isinstance(action, str):
                # Handle string action
                parts = action.split()
                if len(parts) >= 2:
                    direction = parts[0].capitalize()
                    # Extract numeric value with sign
                    amount = "".join([p for p in parts[1:] if p.replace('+', '').replace('-', '').replace('.', '').replace('%', '').isdigit() or p in ['+', '-', '.']])
                    
                    trade_data.append({
                        "Ticker": ticker,
                        "Action": direction,
                        "Change": amount
                    })
        
        if trade_data:
            # Color-code based on action
            def color_action(val):
                if val.lower() in ["buy", "increase", "add"]:
                    return "color: green"
                elif val.lower() in ["sell", "decrease", "reduce"]:
                    return "color: red"
                return ""
            
            trade_df = pd.DataFrame(trade_data)
            st.dataframe(
                trade_df.style.applymap(color_action, subset=["Action"]),
                use_container_width=True
            )
    
    # Display rationale
    if "rationale" in recommendations:
        st.subheader("Rebalancing Rationale")
        st.markdown(recommendations["rationale"])


def display_risk_assessment(results):
    """Display portfolio risk assessment"""
    if "risk_assessment" not in results:
        st.warning("No risk assessment available.")
        return
    
    risk_data = results["risk_assessment"]
    
    # Display overall risk metrics
    st.subheader("Portfolio Risk Profile")
    
    # Create columns for metrics
    cols = st.columns(4)
    
    # Display overall risk rating
    risk_rating = risk_data.get("risk_rating", "Moderate")
    risk_score = risk_data.get("risk_score", 5)
    
    # Normalize risk score to 1-10 if needed
    if isinstance(risk_score, str) and "%" in risk_score:
        risk_score = float(risk_score.strip("%")) / 10
    
    # Determine color based on risk rating
    risk_color = "green" if risk_score < 4 else "orange" if risk_score < 7 else "red"
    
    with cols[0]:
        st.metric("Risk Rating", risk_rating)
        # Create a gauge chart for risk score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risk Score (1-10)"},
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 3.33], "color": "lightgreen"},
                    {"range": [3.33, 6.66], "color": "lightyellow"},
                    {"range": [6.66, 10], "color": "lightcoral"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Display other key metrics
    with cols[1]:
        volatility = risk_data.get("volatility", "--")
        st.metric("Volatility", f"{volatility:.2%}" if isinstance(volatility, (int, float)) else volatility)
    
    with cols[2]:
        sharpe = risk_data.get("sharpe_ratio", "--")
        st.metric("Sharpe Ratio", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe)
    
    with cols[3]:
        diversification = risk_data.get("diversification_score", "--")
        st.metric("Diversification", f"{diversification:.1f}/10" if isinstance(diversification, (int, float)) else diversification)
    
    # Display risk breakdown
    if "sector_exposure" in risk_data:
        st.subheader("Sector Exposure")
        sector_data = risk_data["sector_exposure"]
        
        if isinstance(sector_data, dict):
            # Create horizontal bar chart
            sectors = list(sector_data.keys())
            values = list(sector_data.values())
            
            # Ensure values are numeric
            numeric_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                elif isinstance(v, str) and "%" in v:
                    numeric_values.append(float(v.strip("%")) / 100)
                else:
                    numeric_values.append(0)
            
            fig = px.bar(
                x=numeric_values,
                y=sectors,
                orientation="h",
                title="Sector Allocation",
                color=numeric_values,
                color_continuous_scale="Viridis",
                text=[f"{v:.1%}" for v in numeric_values]
            )
            
            fig.update_layout(xaxis_tickformat=".0%")
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display risk analysis
    if "analysis" in risk_data:
        st.subheader("Risk Analysis")
        st.markdown(risk_data["analysis"])
    
    # Display recommendations
    if "recommendations" in risk_data:
        st.subheader("Risk Reduction Recommendations")
        recommendations = risk_data["recommendations"]
        
        if isinstance(recommendations, list):
            for i, rec in enumerate(recommendations):
                st.markdown(f"{i+1}. {rec}")
        else:
            st.markdown(recommendations)


def display_portfolio_analysis_results():
    """Display portfolio analysis results"""
    if not st.session_state.portfolio_analysis_results:
        return
    
    results = st.session_state.portfolio_analysis_results
    analysis_type = getattr(st.session_state, "analysis_type", "Portfolio Review")
    
    st.header(f"Portfolio Analysis: {analysis_type}")
    
    # Display summary info
    if "summary" in results:
        st.markdown("### Executive Summary")
        st.markdown(results["summary"])
    
    # Based on analysis type, display appropriate visualizations
    if "portfolio_recommendations" in results and analysis_type in ["Rebalancing Recommendations", "Portfolio Review"]:
        display_rebalancing_recommendations(results)
    
    if "risk_assessment" in results and analysis_type in ["Risk Assessment", "Portfolio Review"]:
        display_risk_assessment(results)
    
    # Display individual stock analyses if available
    if any(ticker in results for ticker in st.session_state.portfolio):
        st.subheader("Individual Holdings Analysis")
        
        # Create tabs for each holding
        stock_tabs = st.tabs(list(st.session_state.portfolio.keys()))
        
        for i, ticker in enumerate(st.session_state.portfolio):
            if ticker not in results:
                continue
                
            with stock_tabs[i]:
                ticker_data = results[ticker]
                
                # Display signal and confidence
                signal = ticker_data.get("signal", "neutral")
                confidence = ticker_data.get("confidence", 0.5)
                
                # Color-coded signal
                signal_color = "green" if signal in ["buy", "strong buy", "bullish"] else "red" if signal in ["sell", "strong sell", "bearish"] else "orange"
                st.markdown(f"<h3 style='color: {signal_color};'>Signal: {signal.upper()} (Confidence: {confidence:.1%})</h3>", unsafe_allow_html=True)
                
                # Display reasoning
                st.markdown("### Analysis")
                st.markdown(ticker_data.get("reasoning", "No analysis provided."))
                
                # Display metrics if available
                if "metrics" in ticker_data:
                    st.markdown("### Key Metrics")
                    metrics = ticker_data["metrics"]
                    
                    metrics_data = []
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
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
                    
                    st.table(pd.DataFrame(metrics_data))


def main():
    """Main function for the portfolio management page"""
    st.title("💼 Portfolio Management")
    st.markdown("Optimize and manage your investment portfolio using master-slave agent architecture")
    
    # Create two columns for portfolio and analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Setup")
        display_portfolio_input()
        display_portfolio_summary()
    
    with col2:
        # Create sidebar for analysis settings
        with st.sidebar:
            st.header("Analysis Settings")
            
            # Set the master agent for this page
            if not st.session_state.selected_master_agent:
                st.session_state.selected_master_agent = get_default_master_agent_for_page(CURRENT_PAGE)
            
            # Display the master agent info (read-only)
            st.subheader("Master Agent (Orchestrator)")
            st.info(f"Current Master Agent: Portfolio Master")
            
            # Select slave agents (analysts)
            st.subheader("Select Slave Agents (Analysts)")
            selected_slaves = display_slave_agent_selection(st.session_state.selected_master_agent, multiselect=True)
            st.session_state.selected_slave_agents = selected_slaves
            
            # Display model selection
            display_model_selection()
            
            # Display date range selection
            display_date_range_selection()
            
            # Display run analysis button
            run_portfolio_analysis_button()
        
        # Display portfolio analysis results
        display_portfolio_analysis_results()


if __name__ == "__main__":
    main()
