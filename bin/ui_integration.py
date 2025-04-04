"""
UI Integration for Agno Teams

This module provides utilities for integrating Agno Teams with the Streamlit UI,
allowing seamless selection and execution of master and slave agents.
"""

from typing import Dict, List, Optional, Union, Any
import streamlit as st
from datetime import datetime
import sys
import os
import logging

# Import team coordinator
from bin.team_coordinator import TeamCoordinator, ProgressTracker

# Import master agents
from bin.stock_analysis_master import StockAnalysisMaster
from bin.sector_analysis_master import SectorAnalysisMaster
from bin.portfolio_master import PortfolioMaster

# Import slave agents
from slave_agent.ben_graham_agno import BenGrahamAgnoAgent
from slave_agent.warren_buffett_agno import WarrenBuffettAgnoAgent
from slave_agent.phil_fisher_agno import PhilFisherAgnoAgent
from slave_agent.charlie_munger_agno import CharlieMungerAgnoAgent
from slave_agent.bill_ackman_agno import BillAckmanAgnoAgent
from slave_agent.cathie_wood_agno import CathieWoodAgnoAgent
from slave_agent.stanley_druckenmiller_agno import StanleyDruckenmillerAgnoAgent
from slave_agent.technicals_agno import TechnicalsAgnoAgent
from slave_agent.valuation_agno import ValuationAgnoAgent
from slave_agent.risk_manager_agno import RiskManagerAgnoAgent
from slave_agent.portfolio_manager_agno import PortfolioManagerAgnoAgent
from slave_agent.fundamentals_agno import FundamentalsAgnoAgent
from slave_agent.sentiment_agno import SentimentAgnoAgent

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize team coordinator as a global variable
team_coordinator = TeamCoordinator()


def get_available_master_agents() -> List[Dict[str, Any]]:
    """
    Get list of available master agents
    
    Returns:
        List of agent information dictionaries
    """
    return [
        {
            "id": "stock_analysis_master",
            "name": "Stock Analysis Master",
            "description": "Master agent for comprehensive stock analysis",
            "class": StockAnalysisMaster,
            "category": "stock_analysis"
        },
        {
            "id": "sector_analysis_master",
            "name": "Sector Analysis Master",
            "description": "Master agent for sector-wide analysis",
            "class": SectorAnalysisMaster,
            "category": "sector_analysis"
        },
        {
            "id": "portfolio_master",
            "name": "Portfolio Master",
            "description": "Master agent for portfolio analysis and optimization",
            "class": PortfolioMaster,
            "category": "portfolio_analysis"
        }
    ]


def get_available_slave_agents() -> List[Dict[str, Any]]:
    """
    Get list of available slave agents
    
    Returns:
        List of agent information dictionaries
    """
    return [
        # Investment strategy agents
        {
            "id": "ben_graham",
            "name": "Ben Graham",
            "description": "Value investing specialist",
            "class": BenGrahamAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "value"
        },
        {
            "id": "warren_buffett",
            "name": "Warren Buffett",
            "description": "Value and quality investing specialist",
            "class": WarrenBuffettAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "value_quality"
        },
        {
            "id": "phil_fisher",
            "name": "Phil Fisher",
            "description": "Growth investing specialist",
            "class": PhilFisherAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "growth"
        },
        {
            "id": "charlie_munger",
            "name": "Charlie Munger",
            "description": "Quality investing specialist",
            "class": CharlieMungerAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "quality"
        },
        {
            "id": "bill_ackman",
            "name": "Bill Ackman",
            "description": "Activist investing specialist",
            "class": BillAckmanAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "activist"
        },
        {
            "id": "cathie_wood",
            "name": "Cathie Wood",
            "description": "Innovation investing specialist",
            "class": CathieWoodAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "innovation"
        },
        {
            "id": "stanley_druckenmiller",
            "name": "Stanley Druckenmiller",
            "description": "Macro investing specialist",
            "class": StanleyDruckenmillerAgnoAgent,
            "category": "investment_strategy",
            "analysis_type": "macro"
        },
        
        # Analysis agents
        {
            "id": "technicals",
            "name": "Technicals Agent",
            "description": "Technical analysis specialist",
            "class": TechnicalsAgnoAgent,
            "category": "analysis",
            "analysis_type": "technical"
        },
        {
            "id": "fundamentals",
            "name": "Fundamentals Agent",
            "description": "Fundamental analysis specialist",
            "class": FundamentalsAgnoAgent,
            "category": "analysis",
            "analysis_type": "fundamental"
        },
        {
            "id": "sentiment",
            "name": "Sentiment Agent",
            "description": "Market sentiment analysis specialist",
            "class": SentimentAgnoAgent,
            "category": "analysis",
            "analysis_type": "sentiment"
        },
        {
            "id": "valuation",
            "name": "Valuation Agent",
            "description": "Valuation specialist",
            "class": ValuationAgnoAgent,
            "category": "analysis",
            "analysis_type": "valuation"
        },
        
        # Support agents
        {
            "id": "portfolio_manager",
            "name": "Portfolio Manager",
            "description": "Portfolio management specialist",
            "class": PortfolioManagerAgnoAgent,
            "category": "support",
            "analysis_type": "portfolio"
        },
        {
            "id": "risk_manager",
            "name": "Risk Manager",
            "description": "Risk management specialist",
            "class": RiskManagerAgnoAgent,
            "category": "support",
            "analysis_type": "risk"
        }
    ]


def get_all_available_agents() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Get all available agents grouped by type
    
    Returns:
        Dictionary with agent types and their agents
    """
    master_agents = get_available_master_agents()
    slave_agents = get_available_slave_agents()
    
    # Group slave agents by category
    slave_agents_by_category = {}
    for agent in slave_agents:
        category = agent["category"]
        if category not in slave_agents_by_category:
            slave_agents_by_category[category] = []
        slave_agents_by_category[category].append(agent)
    
    return {
        "master": master_agents,
        "slave": slave_agents_by_category
    }


def get_default_master_agent_for_page(page_name: str = None) -> str:
    """
    Get the default master agent ID for a specific page
    
    Args:
        page_name: Name of the current page
        
    Returns:
        Default master agent ID for the page
    """
    if page_name is None:
        # Try to get the page name from Streamlit
        try:
            import streamlit as st
            page_name = st._config.get_option("_current_page")
        except:
            page_name = ""
    
    # Map page names to default master agents
    page_to_agent_map = {
        "stock_analysis": "stock_analysis_master",
        "1_stock_analysis": "stock_analysis_master",
        "sector_analysis": "sector_analysis_master",
        "2_sector_analysis": "sector_analysis_master",
        "portfolio_analysis": "portfolio_master",
        "3_portfolio_analysis": "portfolio_master"
    }
    
    # Get the default agent ID for this page, or use stock analysis as fallback
    return page_to_agent_map.get(page_name, "stock_analysis_master")


def display_master_agent_selection(page_name: str = None, allow_selection: bool = True) -> str:
    """
    Display master agent selection UI
    
    Args:
        page_name: Name of the current page for automatic selection
        allow_selection: Whether to allow changing the master agent
        
    Returns:
        Selected master agent ID
    """
    # Get all available master agents
    master_agents = get_available_master_agents()
    
    # Create a mapping of agent IDs to their info
    master_agent_map = {agent["id"]: agent for agent in master_agents}
    
    # Get the default master agent for this page
    default_master_agent_id = get_default_master_agent_for_page(page_name)
    
    # Initialize session state for master agent selection if not already present
    if "selected_master_agent" not in st.session_state:
        st.session_state.selected_master_agent = default_master_agent_id
    
    # Display the selection UI if allowed
    if allow_selection:
        # Create a list of agent names for the selectbox
        agent_options = [(agent["id"], f"{agent['name']} - {agent['description']}") 
                        for agent in master_agents]
        
        # Display the selectbox
        selected_option = st.selectbox(
            "Select Master Agent (Coordinator)",
            options=[opt[0] for opt in agent_options],
            format_func=lambda x: next((opt[1] for opt in agent_options if opt[0] == x), x),
            index=[i for i, opt in enumerate(agent_options) if opt[0] == st.session_state.selected_master_agent][0]
        )
        
        # Update the session state
        st.session_state.selected_master_agent = selected_option
    else:
        # Just display the selected agent without allowing changes
        selected_agent = master_agent_map[st.session_state.selected_master_agent]
        st.info(f"Using {selected_agent['name']} for analysis")
    
    return st.session_state.selected_master_agent


def display_agent_type_selection() -> str:
    """
    Display agent type selection UI
    
    Returns:
        Selected agent type
    """
    # Get all available agents grouped by type
    all_agents = get_all_available_agents()
    
    # Get the slave agent categories
    slave_categories = list(all_agents["slave"].keys())
    
    # Create a mapping for display names
    category_display_names = {
        "investment_strategy": "Investment Strategy Agents",
        "analysis": "Analysis Agents",
        "support": "Support Agents"
    }
    
    # Initialize session state for agent type selection if not already present
    if "selected_agent_type" not in st.session_state:
        st.session_state.selected_agent_type = slave_categories[0] if slave_categories else ""
    
    # Display the selection UI
    selected_option = st.selectbox(
        "Select Agent Type",
        options=slave_categories,
        format_func=lambda x: category_display_names.get(x, x),
        index=slave_categories.index(st.session_state.selected_agent_type) if st.session_state.selected_agent_type in slave_categories else 0
    )
    
    # Update the session state
    st.session_state.selected_agent_type = selected_option
    
    return st.session_state.selected_agent_type


def display_slave_agent_selection(master_agent_id: str, multiselect: bool = True) -> List[str]:
    """
    Display slave agent selection UI
    
    Args:
        master_agent_id: ID of the selected master agent
        multiselect: Whether to allow selecting multiple slave agents
        
    Returns:
        List of selected slave agent IDs
    """
    # Get all available agents grouped by type
    all_agents = get_all_available_agents()
    
    # Get the slave agent categories
    slave_categories = list(all_agents["slave"].keys())
    
    # Create a mapping for display names
    category_display_names = {
        "investment_strategy": "Investment Strategy Agents",
        "analysis": "Analysis Agents",
        "support": "Support Agents"
    }
    
    # Initialize session state for slave agent selection if not already present
    if "selected_slave_agents" not in st.session_state:
        st.session_state.selected_slave_agents = {}
    
    # Initialize the session state for this master agent if not already present
    if master_agent_id not in st.session_state.selected_slave_agents:
        st.session_state.selected_slave_agents[master_agent_id] = []
    
    # Create tabs for agent categories
    if slave_categories:
        tabs = st.tabs([category_display_names.get(cat, cat) for cat in slave_categories])
        
        for i, category in enumerate(slave_categories):
            with tabs[i]:
                agents_in_category = all_agents["slave"][category]
                
                # Create a list of agent names for the multiselect
                agent_options = [(agent["id"], f"{agent['name']} - {agent['description']}") 
                                for agent in agents_in_category]
                
                # Display the multiselect or select based on the multiselect parameter
                if multiselect:
                    selected_options = st.multiselect(
                        f"Select {category_display_names.get(category, category)}",
                        options=[opt[0] for opt in agent_options],
                        default=[
                            opt[0] for opt in agent_options 
                            if opt[0] in st.session_state.selected_slave_agents[master_agent_id]
                        ],
                        format_func=lambda x: next((opt[1] for opt in agent_options if opt[0] == x), x)
                    )
                else:
                    # If not multiselect, use radio buttons
                    selected_option = st.radio(
                        f"Select {category_display_names.get(category, category)}",
                        options=["None"] + [opt[0] for opt in agent_options],
                        index=0,
                        format_func=lambda x: "None" if x == "None" else next((opt[1] for opt in agent_options if opt[0] == x), x)
                    )
                    
                    selected_options = [selected_option] if selected_option != "None" else []
                
                # Update the session state for this category
                for agent in agents_in_category:
                    agent_id = agent["id"]
                    if agent_id in selected_options and agent_id not in st.session_state.selected_slave_agents[master_agent_id]:
                        st.session_state.selected_slave_agents[master_agent_id].append(agent_id)
                    elif agent_id not in selected_options and agent_id in st.session_state.selected_slave_agents[master_agent_id]:
                        st.session_state.selected_slave_agents[master_agent_id].remove(agent_id)
    
    # Display the selected agents
    if st.session_state.selected_slave_agents[master_agent_id]:
        st.write("Selected Agents:")
        
        # Get all slave agents
        all_slave_agents = get_available_slave_agents()
        slave_agent_map = {agent["id"]: agent for agent in all_slave_agents}
        
        # Display the selected agents
        for agent_id in st.session_state.selected_slave_agents[master_agent_id]:
            if agent_id in slave_agent_map:
                agent = slave_agent_map[agent_id]
                st.write(f"- {agent['name']} ({agent['description']})")
    else:
        st.warning("No slave agents selected. Please select at least one agent.")
    
    return st.session_state.selected_slave_agents[master_agent_id]


def create_team_for_analysis(master_agent_id: str, slave_agent_ids: List[str], model_provider: str, model_name: str):
    """
    Create a team for analysis using the selected agents
    
    Args:
        master_agent_id: ID of the master agent
        slave_agent_ids: IDs of the slave agents
        model_provider: LLM provider
        model_name: LLM model name
        
    Returns:
        Created team
    """
    # Get agent information
    master_agents = get_available_master_agents()
    slave_agents = get_available_slave_agents()
    
    # Create mappings for easier lookup
    master_agent_map = {agent["id"]: agent for agent in master_agents}
    slave_agent_map = {agent["id"]: agent for agent in slave_agents}
    
    # Check if the master agent exists
    if master_agent_id not in master_agent_map:
        raise ValueError(f"Master agent {master_agent_id} not found")
    
    # Get the master agent info
    master_agent_info = master_agent_map[master_agent_id]
    
    # Create the master agent instance
    master_agent_class = master_agent_info["class"]
    master_agent_instance = master_agent_class(model_name=model_name, model_provider=model_provider)
    
    # Create slave agent instances
    slave_agent_instances = {}
    slave_agent_info_list = []
    
    for agent_id in slave_agent_ids:
        if agent_id in slave_agent_map:
            info = slave_agent_map[agent_id]
            agent_class = info["class"]
            agent_instance = agent_class(model_name=model_name, model_provider=model_provider)
            
            slave_agent_instances[agent_id] = {
                "instance": agent_instance,
                "info": info
            }
            
            slave_agent_info_list.append(info)
    
    # Get model configuration from llm_manager
    from llm.providers import llm_manager
    
    # Get the model configuration using llm_manager
    model_config = llm_manager.get_provider_model(model_provider, model_name)
    
    # Create Agno agents for team members
    from bin.team_coordinator import Agent as AgnoAgent
    
    # Create the coordinator agent
    coordinator = AgnoAgent(
        name=master_agent_info["name"],
        role=f"Coordinator for {master_agent_info['description']}",
        model_name=model_name,
        model_provider=model_provider,
        instructions=[
            "You are the coordinator for a financial analysis team.",
            "Your job is to delegate tasks to specialized agents and synthesize their results.",
            "Break down the analysis into subtasks and assign them to the appropriate team members.",
            "After receiving all results, synthesize them into a comprehensive analysis report."
        ]
    )
    
    # Create slave agents list
    team_members = [coordinator.agent]
    
    # Create slave agents
    for agent_id, agent_data in slave_agent_instances.items():
        agent_info = agent_data["info"]
        
        slave_agent = AgnoAgent(
            name=agent_info["name"],
            role=agent_info["description"],
            model_name=model_name,
            model_provider=model_provider,
            instructions=[
                f"You are a specialist in {agent_info['analysis_type']} analysis.",
                "Analyze the given data according to your specialty and provide detailed insights.",
                "Focus on providing actionable recommendations based on your analysis.",
                "Include confidence scores and reasoning for your recommendations."
            ]
        )
        
        # Add to members list
        team_members.append(slave_agent.agent)
    
    # Create team with members only (no model)
    team_name = f"{master_agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Use the custom Team wrapper from agno_wrapper.py instead of directly using Agno Team
    from bin.agno_wrapper import Team as AgnoTeamWrapper
    
    # Print debug info to console directly
    print("\n==== DEBUG INFO =====")
    print(f"Creating team with name: {team_name}")
    print(f"Team members count: {len(team_members)}")
    
    try:
        # Create the team using the custom wrapper
        print("Creating Team using custom AgnoTeamWrapper")
        team = AgnoTeamWrapper(
            name=team_name,
            description=f"Team for {master_agent_info['name']} analysis",
            mode="coordinate",
            members=team_members,
            show_members_responses=True,
            markdown=True
        )
        print("Team created successfully using custom wrapper")
        
    except Exception as e:
        print(f"ERROR creating team: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Register the team with the coordinator
    team_coordinator.teams[team_name] = team
    team_coordinator.logger.info(f"Created team {team_name} with coordinate mode")
    
    return team


def run_analysis_with_teams(tickers: List[str],
                          master_agent_id: str,
                          slave_agent_ids: List[str],
                          provider: str,
                          model: str,
                          start_date: datetime,
                          end_date: datetime,
                          is_sector_analysis: bool = False,
                          sector: str = None,
                          is_portfolio_analysis: bool = False,
                          portfolio_data: Dict[str, Any] = None):
    """
    Run financial analysis using master-slave teams
    
    Args:
        tickers: List of stock tickers to analyze
        master_agent_id: ID of the master agent
        slave_agent_ids: List of slave agent IDs
        provider: LLM provider
        model: LLM model name
        start_date: Start date for analysis
        end_date: End date for analysis
        is_sector_analysis: Whether this is a sector analysis
        sector: Selected sector for sector analysis
        is_portfolio_analysis: Whether this is a portfolio analysis
        portfolio_data: Portfolio data for portfolio analysis
        
    Returns:
        Analysis results
    """
    # Create progress tracker
    progress = ProgressTracker(total_steps=5, name="TeamAnalysis")
    progress.start()
    
    try:
        # Create the team
        progress.step("Creating team")
        team = create_team_for_analysis(
            master_agent_id=master_agent_id,
            slave_agent_ids=slave_agent_ids,
            model_provider=provider,
            model_name=model
        )
        
        # Prepare the analysis prompt based on the type of analysis
        progress.step("Preparing analysis prompt")
        
        if is_sector_analysis:
            prompt = f"""Perform a comprehensive sector analysis for the {sector} sector.
            Focus on the following tickers: {', '.join(tickers)}.
            Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
            
            Provide the following in your analysis:
            1. Overall sector health and trends
            2. Comparative analysis of the specified companies
            3. Key performance indicators and metrics
            4. Investment recommendations with confidence scores
            5. Risk assessment and outlook
            
            Coordinate with your team members to ensure a comprehensive analysis.
            """
        elif is_portfolio_analysis:
            # Format portfolio data for the prompt
            portfolio_str = "\n".join([f"- {ticker}: {details['weight']}% allocation" for ticker, details in portfolio_data.items()])
            
            prompt = f"""Perform a comprehensive portfolio analysis for the following portfolio:
            {portfolio_str}
            
            Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
            
            Provide the following in your analysis:
            1. Overall portfolio performance and health
            2. Individual asset analysis
            3. Risk assessment (volatility, drawdowns, etc.)
            4. Diversification analysis
            5. Rebalancing recommendations with confidence scores
            6. Future outlook and strategy recommendations
            
            Coordinate with your team members to ensure a comprehensive analysis.
            """
        else:
            # Regular stock analysis
            prompt = f"""Perform a comprehensive analysis for {', '.join(tickers)}.
            Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.
            
            Provide the following in your analysis:
            1. Company overview and business model
            2. Financial performance analysis
            3. Valuation assessment
            4. Technical analysis
            5. Market sentiment and news impact
            6. Investment recommendation with confidence score
            7. Risk assessment and outlook
            
            Coordinate with your team members to ensure a comprehensive analysis.
            """
        
        # Run the team analysis
        progress.step("Running team analysis")
        logger.info(f"Running analysis with teams: {master_agent_id} + {len(slave_agent_ids)} slave agents")
        
        # Run the team analysis with the specified model configuration
        result = team.run(
            prompt, 
            show_members_responses=True, 
            markdown=True
        )
        
        # Process the results based on the type of analysis
        progress.step("Processing results")
        
        if is_sector_analysis:
            processed_result = _process_sector_analysis_result(result)
        elif is_portfolio_analysis:
            processed_result = _process_portfolio_analysis_result(result)
        else:
            processed_result = _process_stock_analysis_result(result)
        
        # Complete the progress tracking
        progress.complete("Analysis completed successfully")
        
        return processed_result
    
    except Exception as e:
        # Log the error and mark progress as failed
        logger.error(f"Error in team analysis: {str(e)}")
        progress.fail(f"Error in analysis: {str(e)}")
        
        # Return a basic error result
        return {
            "status": "error",
            "error": str(e),
            "tickers": tickers
        }


def _process_stock_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process stock analysis result into UI-friendly format
    
    Args:
        result: Raw result from team
        
    Returns:
        Processed result
    """
    # Extract the main content from the result
    content = result.get("response", "")
    
    # Create a basic processed result
    processed = {
        "content": content,
        "team_responses": result.get("members_responses", {}),
        "status": "success"
    }
    
    # Try to extract structured data if available
    try:
        # Look for investment recommendation section
        import re
        
        # Extract recommendation if present
        recommendation_pattern = r"Investment\s+Recommendation[:\s]+(Buy|Sell|Hold|Strong Buy|Strong Sell|Neutral)\s"
        recommendation_match = re.search(recommendation_pattern, content, re.IGNORECASE)
        
        if recommendation_match:
            processed["recommendation"] = recommendation_match.group(1)
        
        # Extract confidence score if present
        confidence_pattern = r"Confidence\s+Score[:\s]+(\d+(\.\d+)?)%?"
        confidence_match = re.search(confidence_pattern, content, re.IGNORECASE)
        
        if confidence_match:
            processed["confidence"] = float(confidence_match.group(1))
        
        # Extract key metrics if present
        metrics = {}
        
        # Common financial metrics to look for
        metric_patterns = {
            "pe_ratio": r"P/E\s+Ratio[:\s]+(\d+(\.\d+)?)",
            "pb_ratio": r"P/B\s+Ratio[:\s]+(\d+(\.\d+)?)",
            "dividend_yield": r"Dividend\s+Yield[:\s]+(\d+(\.\d+)?)%?",
            "roe": r"ROE[:\s]+(\d+(\.\d+)?)%?",
            "roa": r"ROA[:\s]+(\d+(\.\d+)?)%?"
        }
        
        for metric_key, pattern in metric_patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[metric_key] = float(match.group(1))
        
        if metrics:
            processed["metrics"] = metrics
    
    except Exception as e:
        # If there's an error in extraction, just log it and continue
        logger.warning(f"Error extracting structured data from analysis: {str(e)}")
    
    return processed


def _process_sector_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process sector analysis result into UI-friendly format
    
    Args:
        result: Raw result from team
        
    Returns:
        Processed result
    """
    # Extract the main content from the result
    content = result.get("response", "")
    
    # Create a basic processed result
    processed = {
        "content": content,
        "team_responses": result.get("members_responses", {}),
        "status": "success"
    }
    
    # Try to extract structured data if available
    try:
        # Look for sector outlook section
        import re
        
        # Extract sector outlook if present
        outlook_pattern = r"Sector\s+Outlook[:\s]+(Positive|Negative|Neutral|Bullish|Bearish|Mixed)\s"
        outlook_match = re.search(outlook_pattern, content, re.IGNORECASE)
        
        if outlook_match:
            processed["sector_outlook"] = outlook_match.group(1)
        
        # Extract top performers if present
        top_performers = []
        top_performers_section = re.search(r"Top\s+Performers[:\s]+([\s\S]+?)(?:\n\n|\n#|$)", content, re.IGNORECASE)
        
        if top_performers_section:
            # Extract ticker symbols from the section
            tickers = re.findall(r"\b[A-Z]{1,5}\b", top_performers_section.group(1))
            top_performers = list(set(tickers))  # Remove duplicates
        
        if top_performers:
            processed["top_performers"] = top_performers
    
    except Exception as e:
        # If there's an error in extraction, just log it and continue
        logger.warning(f"Error extracting structured data from sector analysis: {str(e)}")
    
    return processed


def _process_portfolio_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process portfolio analysis result into UI-friendly format
    
    Args:
        result: Raw result from team
        
    Returns:
        Processed result
    """
    # Extract the main content from the result
    content = result.get("response", "")
    
    # Create a basic processed result
    processed = {
        "content": content,
        "team_responses": result.get("members_responses", {}),
        "status": "success"
    }
    
    # Try to extract structured data if available
    try:
        # Look for portfolio metrics section
        import re
        
        # Extract portfolio risk if present
        risk_pattern = r"Portfolio\s+Risk[:\s]+(Low|Medium|High|Very Low|Very High)\s"
        risk_match = re.search(risk_pattern, content, re.IGNORECASE)
        
        if risk_match:
            processed["portfolio_risk"] = risk_match.group(1)
        
        # Extract diversification score if present
        diversification_pattern = r"Diversification\s+Score[:\s]+(\d+(\.\d+)?)\s*/\s*10"
        diversification_match = re.search(diversification_pattern, content, re.IGNORECASE)
        
        if diversification_match:
            processed["diversification_score"] = float(diversification_match.group(1))
        
        # Extract rebalancing recommendations if present
        rebalance_section = re.search(r"Rebalancing\s+Recommendations[:\s]+([\s\S]+?)(?:\n\n|\n#|$)", content, re.IGNORECASE)
        
        if rebalance_section:
            processed["rebalance_recommendations"] = rebalance_section.group(1).strip()
    
    except Exception as e:
        # If there's an error in extraction, just log it and continue
        logger.warning(f"Error extracting structured data from portfolio analysis: {str(e)}")
    
    return processed


def initialize_ui_integration():
    """
    Initialize UI integration for Agno Teams
    
    This function should be called when the application starts to enable
    the integration of Agno Teams with the Streamlit UI.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger.info("Initialized UI integration for Agno Teams")
