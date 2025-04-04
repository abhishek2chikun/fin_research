"""
Sector Analysis Page for the Financial Agent UI.
This page provides sector analysis using the master-slave agent architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

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
    page_title="Sector Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

if 'selected_master_agent' not in st.session_state:
    st.session_state.selected_master_agent = None  # Will be set automatically based on page

if 'selected_slave_agents' not in st.session_state:
    st.session_state.selected_slave_agents = []

if 'sector_analysis_results' not in st.session_state:
    st.session_state.sector_analysis_results = {}

if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = LLMProvider.LMSTUDIO.value

if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "sufe-aiflm-lab_fin-r1"

if 'date_range' not in st.session_state:
    st.session_state.date_range = (datetime.now() - timedelta(days=365), datetime.now())

if 'team_analysis_mode' not in st.session_state:
    st.session_state.team_analysis_mode = True
    
# Set the current page for master agent selection
CURRENT_PAGE = "sector_analysis"


def display_sector_selection():
    """Display sector selection widget"""
    sectors = load_available_sectors()
    sector = st.selectbox(
        "Select a sector to analyze",
        options=[""] + sectors,
        format_func=lambda x: "Select a sector..." if x == "" else x,
        key="sector_selection"
    )
    
    if sector and sector != st.session_state.selected_sector:
        st.session_state.selected_sector = sector
        # Clear previous selections when sector changes
        st.session_state.selected_tickers = []
        st.session_state.sector_analysis_results = {}


def display_sector_tickers_selection():
    """Display ticker selection for the chosen sector"""
    if not st.session_state.selected_sector:
        return
    
    # Get tickers for the selected sector
    sector_tickers = load_available_tickers(sector=st.session_state.selected_sector)
    
    if not sector_tickers:
        st.warning(f"No tickers found for sector: {st.session_state.selected_sector}")
        return
    
    # Allow selecting multiple tickers
    selected_tickers = st.multiselect(
        f"Select tickers in {st.session_state.selected_sector} sector",
        options=sector_tickers,
        key="sector_tickers_selection",
        default=st.session_state.selected_tickers if st.session_state.selected_tickers else None
    )
    
    st.session_state.selected_tickers = selected_tickers


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


def run_sector_analysis_button():
    """Display run analysis button for team-based sector analysis"""
    if not st.session_state.selected_sector:
        st.warning("Please select a sector first.")
        return
    
    if not st.session_state.selected_tickers or len(st.session_state.selected_tickers) < 2:
        st.warning("Please select at least two tickers to compare within the sector.")
        return
    
    if not st.session_state.selected_master_agent:
        st.warning("Please select a master agent.")
        return
    
    if not st.session_state.selected_slave_agents:
        st.warning("Please select at least one slave agent.")
        return
    
    if st.button("Run Sector Analysis"):
        with st.spinner(f"Running sector analysis for {st.session_state.selected_sector}... This may take several minutes."):
            try:
                # Run analysis using Agno Teams
                results = run_analysis_with_teams(
                    tickers=st.session_state.selected_tickers,
                    master_agent_id=st.session_state.selected_master_agent,
                    slave_agent_ids=st.session_state.selected_slave_agents,
                    provider=st.session_state.llm_provider,
                    model=st.session_state.llm_model,
                    start_date=st.session_state.date_range[0],
                    end_date=st.session_state.date_range[1],
                    is_sector_analysis=True,
                    sector=st.session_state.selected_sector,
                    is_portfolio_analysis=False
                )
                
                # Store results in session state
                st.session_state.sector_analysis_results = results
                
                # Log completion
                logger.info(f"Sector analysis completed for {st.session_state.selected_sector} with tickers {st.session_state.selected_tickers}")
                
                # Force rerun to display results
                st.rerun()
            except Exception as e:
                st.error(f"Error in sector analysis: {str(e)}")
                logger.error(f"Error in sector analysis: {str(e)}")


def display_stock_comparison_chart(results):
    """Display comparative chart for stocks in the sector"""
    if not results or not isinstance(results, dict):
        return
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for ticker, ticker_data in results.items():
        # Extract metrics for comparison
        if "metrics" in ticker_data:
            metrics = ticker_data["metrics"]
            
            # Only add key metrics
            data_row = {"Ticker": ticker}
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    data_row[metric_name] = metric_value
            
            comparison_data.append(data_row)
    
    if not comparison_data:
        return
    
    # Create dataframe
    df = pd.DataFrame(comparison_data)
    
    # Select visualization metrics
    if len(df.columns) > 2:  # Need at least one metric beyond Ticker
        st.subheader("Comparative Analysis")
        
        # Allow user to select metrics for visualization
        numeric_columns = [col for col in df.columns if col != "Ticker" and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_columns:
            return
            
        # Select top 4 metrics by default (or all if fewer than 4)
        default_metrics = numeric_columns[:min(4, len(numeric_columns))]
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            options=numeric_columns,
            default=default_metrics
        )
        
        if not selected_metrics:
            return
            
        # Create visualization
        fig = go.Figure()
        
        # Determine chart type based on number of stocks and metrics
        if len(selected_metrics) == 1:
            # Bar chart for single metric
            fig = px.bar(
                df, 
                x="Ticker", 
                y=selected_metrics[0],
                title=f"Comparison of {selected_metrics[0]} across stocks",
                color="Ticker"
            )
        else:
            # Radar chart for multiple metrics
            for ticker in df["Ticker"].unique():
                ticker_data = df[df["Ticker"] == ticker]
                
                fig.add_trace(go.Scatterpolar(
                    r=[ticker_data[metric].values[0] for metric in selected_metrics],
                    theta=selected_metrics,
                    fill='toself',
                    name=ticker
                ))
                
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                title="Multi-dimensional Comparison"
            )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)


def display_sector_ranking(results):
    """Display ranking of stocks within the sector"""
    if not results or not isinstance(results, dict):
        return
    
    # Extract signals and confidence scores
    ranking_data = []
    
    for ticker, ticker_data in results.items():
        signal = ticker_data.get("signal", "neutral")
        confidence = ticker_data.get("confidence", 0.5)
        
        # Map signal to numeric score
        signal_score = 0
        if signal.lower() in ["buy", "strong buy", "bullish"]:
            signal_score = confidence
        elif signal.lower() in ["sell", "strong sell", "bearish"]:
            signal_score = -confidence
        
        ranking_data.append({
            "Ticker": ticker,
            "Signal": signal.upper(),
            "Confidence": confidence,
            "Score": signal_score
        })
    
    if not ranking_data:
        return
    
    # Create dataframe and sort by score
    df = pd.DataFrame(ranking_data)
    df = df.sort_values("Score", ascending=False)
    
    # Display ranking
    st.subheader("Sector Ranking")
    
    # Create a colorful table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Rank", "Ticker", "Signal", "Confidence", "Score"],
            fill_color='royalblue',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[
                list(range(1, len(df) + 1)),
                df["Ticker"],
                df["Signal"],
                [f"{x:.1%}" for x in df["Confidence"]],
                [f"{x:.2f}" for x in df["Score"]]
            ],
            fill_color=[
                ['whitesmoke'] * len(df),
                ['whitesmoke'] * len(df),
                [
                    'lightgreen' if x.lower() in ["buy", "strong buy", "bullish"] else 
                    'lightcoral' if x.lower() in ["sell", "strong sell", "bearish"] else 
                    'lightyellow' 
                    for x in df["Signal"]
                ],
                ['whitesmoke'] * len(df),
                [
                    'lightgreen' if x > 0.2 else
                    'lightcoral' if x < -0.2 else
                    'lightyellow'
                    for x in df["Score"]
                ]
            ],
            align='center'
        )
    )])
    
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        height=min(100 + 25 * len(df), 500)  # Adjust height based on number of rows
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_sector_summary(results, sector):
    """Display overall sector summary"""
    if not results or not isinstance(results, dict):
        return
    
    # Calculate overall sector sentiment
    signals = []
    for ticker, ticker_data in results.items():
        signal = ticker_data.get("signal", "neutral")
        confidence = ticker_data.get("confidence", 0.5)
        
        # Map signal to numeric score
        if signal.lower() in ["buy", "strong buy", "bullish"]:
            signals.append(confidence)
        elif signal.lower() in ["sell", "strong sell", "bearish"]:
            signals.append(-confidence)
        else:
            signals.append(0)
    
    if not signals:
        return
    
    # Calculate average sentiment
    avg_signal = sum(signals) / len(signals)
    
    # Determine overall signal
    if avg_signal > 0.3:
        overall_signal = "BULLISH"
        signal_color = "green"
    elif avg_signal < -0.3:
        overall_signal = "BEARISH"
        signal_color = "red"
    else:
        overall_signal = "NEUTRAL"
        signal_color = "orange"
    
    # Display sector summary
    st.header(f"{sector} Sector Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"<h2 style='color: {signal_color};'>Overall: {overall_signal}</h2>", unsafe_allow_html=True)
        st.metric("Analyzed Stocks", len(results))
        
        # Count recommendations
        bullish_count = sum(1 for s in signals if s > 0.1)
        bearish_count = sum(1 for s in signals if s < -0.1)
        neutral_count = len(signals) - bullish_count - bearish_count
        
        # Create a simple pie chart of recommendations
        fig = px.pie(
            values=[bullish_count, neutral_count, bearish_count],
            names=["Bullish", "Neutral", "Bearish"],
            color_discrete_sequence=["green", "orange", "red"],
            title="Signal Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Extract overall sector reasoning if available
        if "sector_summary" in next(iter(results.values()), {}):
            sector_summary = next(iter(results.values()))["sector_summary"]
            st.markdown("### Sector Outlook")
            st.markdown(sector_summary)
        else:
            # Generate a simple sector summary
            st.markdown("### Sector Summary")
            st.markdown(f"Analysis of {len(results)} stocks in the {sector} sector.")
            st.markdown(f"Overall sector sentiment: **{overall_signal}**")
            st.markdown(f"- {bullish_count} stocks with bullish signals")
            st.markdown(f"- {neutral_count} stocks with neutral signals")
            st.markdown(f"- {bearish_count} stocks with bearish signals")


def display_sector_analysis_results():
    """Display sector analysis results"""
    if not st.session_state.sector_analysis_results:
        return
    
    results = st.session_state.sector_analysis_results
    
    # Display sector summary
    display_sector_summary(results, st.session_state.selected_sector)
    
    # Display sector ranking
    display_sector_ranking(results)
    
    # Display comparison chart
    display_stock_comparison_chart(results)
    
    # Create tabs for each ticker in the sector
    st.subheader("Individual Stock Analysis")
    ticker_tabs = st.tabs(list(results.keys()))
    
    # Display detailed analysis for each ticker
    for i, (ticker, ticker_data) in enumerate(results.items()):
        with ticker_tabs[i]:
            # Display signal and confidence
            signal = ticker_data.get("signal", "neutral")
            confidence = ticker_data.get("confidence", 0.5)
            
            # Color-coded signal
            signal_color = "green" if signal in ["buy", "strong buy", "bullish"] else "red" if signal in ["sell", "strong sell", "bearish"] else "orange"
            st.markdown(f"<h3 style='color: {signal_color};'>Signal: {signal.upper()} (Confidence: {confidence:.1%})</h3>", unsafe_allow_html=True)
            
            # Create columns for strengths and weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Strengths")
                strengths = ticker_data.get("strengths", [])
                for strength in strengths:
                    st.markdown(f"✅ {strength}")
            
            with col2:
                st.markdown("#### Weaknesses")
                weaknesses = ticker_data.get("weaknesses", [])
                for weakness in weaknesses:
                    st.markdown(f"⚠️ {weakness}")
            
            # Display reasoning
            st.markdown("#### Analysis")
            st.markdown(ticker_data.get("reasoning", "No reasoning provided."))
            
            # Display metrics if available
            if "metrics" in ticker_data:
                st.markdown("#### Key Metrics")
                metrics = ticker_data["metrics"]
                
                # Format metrics for display
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
                    
                    metrics_data.append({
                        "Metric": key.replace("_", " ").title(),
                        "Value": formatted_value
                    })
                
                st.table(pd.DataFrame(metrics_data))


def main():
    """Main function for the sector analysis page"""
    st.title("📊 Sector Analysis")
    st.markdown("Compare and analyze multiple stocks within a sector using master-slave agent architecture")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Sector Analysis Settings")
        
        # Display sector selection
        display_sector_selection()
        
        # Display ticker selection for sector
        display_sector_tickers_selection()
        
        # Set the master agent for this page
        if not st.session_state.selected_master_agent:
            st.session_state.selected_master_agent = get_default_master_agent_for_page(CURRENT_PAGE)
        
        # Display the master agent info (read-only)
        st.subheader("Master Agent (Orchestrator)")
        st.info(f"Current Master Agent: Sector Analysis Master")
        
        # Select slave agents (analysts)
        st.subheader("Select Slave Agents (Analysts)")
        selected_slaves = display_slave_agent_selection(st.session_state.selected_master_agent, multiselect=True)
        st.session_state.selected_slave_agents = selected_slaves
        
        # Display model selection
        display_model_selection()
        
        # Display date range selection
        display_date_range_selection()
        
        # Display run analysis button
        run_sector_analysis_button()
    
    # Display sector analysis results
    display_sector_analysis_results()


if __name__ == "__main__":
    main()
