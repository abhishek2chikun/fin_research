"""
Stock Analysis Master Agent

This module implements a master agent that orchestrates multiple slave agents 
for comprehensive stock analysis using Agno Teams with the coordinate mode.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import logging
from datetime import datetime
import json

# Import team coordinator components
from bin.team_coordinator import Agent, Tool, ProgressTracker, ProgressStatus

# Import Agno Teams components
import agno
from agno.team import Team

# Import data models
from data.fundamental_data import FinancialData
from data.ohlcv import OHLCVData


class StockAnalysisMaster:
    """
    Master agent for orchestrating stock analysis using Agno Teams coordinate mode
    
    This agent coordinates multiple slave agents to perform comprehensive
    stock analysis and synthesizes their results into a final report.
    It uses the Agno Teams coordinate mode to manage the workflow.
    """
    
    def __init__(self, 
               model_name: str = "gpt-4", 
               model_provider: str = "openai",
               analysis_types: List[str] = None):
        """
        Initialize the stock analysis master agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
            analysis_types: Types of analysis to perform
        """
        self.model_name = model_name
        self.model_provider = model_provider
        
        # Set up analysis types
        self.analysis_types = analysis_types or [
            "value", "growth", "quality", "technical", "sentiment", "valuation"
        ]
        
        # Initialize progress tracker
        self.progress = ProgressTracker(
            total_steps=len(self.analysis_types) + 2,  # Analysis types + data prep + final report
            name="StockAnalysis"
        )
        
        # Initialize team coordination components
        self.team = None
        self.agent = None
        self.tools = []
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set up tools
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools for the agent"""
        
        # Tool for getting fundamental data
        get_fundamental_data_tool = Tool(
            name="get_fundamental_data",
            func=self._get_fundamental_data,
            description="Get fundamental financial data for a stock"
        )
        
        # Tool for getting OHLCV data
        get_ohlcv_data_tool = Tool(
            name="get_ohlcv_data",
            func=self._get_ohlcv_data,
            description="Get OHLCV price data for a stock"
        )
        
        # Tool for delegating analysis to slave agents
        delegate_analysis_tool = Tool(
            name="delegate_analysis",
            func=self._delegate_analysis,
            description="Delegate analysis to a slave agent"
        )
        
        # Tool for aggregating results
        aggregate_results_tool = Tool(
            name="aggregate_results",
            func=self._aggregate_results_tool,
            description="Aggregate results from multiple slave agents"
        )
        
        # Add tools to the list
        self.tools = [
            get_fundamental_data_tool,
            get_ohlcv_data_tool,
            delegate_analysis_tool,
            aggregate_results_tool
        ]
    
    def _get_fundamental_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Fundamental data
        """
        self.logger.info(f"Getting fundamental data for {ticker}")
        
        # Implement the logic to get fundamental data
        # This is a placeholder - in a real implementation, you would use
        # a data provider or load from a database
        
        # For now, return a placeholder
        return {"ticker": ticker, "data_available": True}
    
    def _get_ohlcv_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Get OHLCV data for a stock
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data
            
        Returns:
            OHLCV data
        """
        self.logger.info(f"Getting OHLCV data for {ticker} ({period})")
        
        # Implement the logic to get OHLCV data
        # This is a placeholder - in a real implementation, you would use
        # a data provider or load from a database
        
        # For now, return a placeholder
        return {"ticker": ticker, "period": period, "data_available": True}
    
    def setup_team(self, slave_agents: Dict[str, Any]):
        """
        Set up the Agno Team with coordinate mode for stock analysis
        
        Args:
            slave_agents: Dictionary of slave agents to include in the team
        """
        self.logger.debug(f"Setting up team with {len(slave_agents)} slave agents")
        self.logger.debug(f"Model name: {self.model_name}, Provider: {self.model_provider}")
        # Get the model configuration from the provider manager - only for agents, not for team
        from llm.providers import llm_manager
        try:
            # We'll get the model for individual agents, but NOT for the team
            agent_model = llm_manager.get_provider_model(self.model_provider, self.model_name)
            self.logger.debug(f"Got model from provider manager for agents: {type(agent_model)}")
        except Exception as e:
            self.logger.error(f"Error getting model from provider manager: {e}")
            # Fallback to not using a model object
            agent_model = None
        
        # Create the coordinator agent
        self.logger.debug("Creating coordinator agent")
        try:
            coordinator = Agent(
                name="Stock Analysis Coordinator",
                role="Coordinator for stock analysis",
                model_name=self.model_name,
                model_provider=self.model_provider,
                instructions=[
                    "You are the coordinator for a stock analysis team.",
                    "Your job is to delegate tasks to specialized agents and synthesize their results.",
                    "Break down the analysis into subtasks and assign them to the appropriate team members.",
                    "After receiving all results, synthesize them into a comprehensive analysis."
                ]
            )
            self.logger.debug("Successfully created coordinator agent")
        except Exception as e:
            self.logger.error(f"Error creating coordinator agent: {e}")
            raise
        
        # Create slave agents list
        team_members = [coordinator]
        
        # Process slave agents
        for agent_id, info in slave_agents.items():
            agent_instance = info["instance"]
            analysis_type = agent_instance.analysis_type if hasattr(agent_instance, 'analysis_type') else 'financial'
            
            # Create Agno agent for the slave
            slave_agent = Agent(
                name=f"{agent_id.replace('_', ' ').title()}",
                role=f"Specialist in {analysis_type} analysis",
                model_name=self.model_name,
                model_provider=self.model_provider,
                instructions=[
                    f"You are a specialist in {analysis_type} analysis.",
                    "Analyze the given data according to your specialty and provide detailed insights.",
                    "Focus on identifying key patterns, trends, and actionable recommendations."
                ]
            )
            
            # Add tools from the slave agent instance if available
            if hasattr(agent_instance, 'tools') and agent_instance.tools:
                for tool in agent_instance.tools:
                    slave_agent.add_tool(tool)
            
            # Add to team members list
            team_members.append(slave_agent)
        
        # Create a new team for stock analysis with coordinate mode and all members
        self.logger.debug(f"Creating team with {len(team_members)} members")
        try:
            # Create the team WITHOUT passing the model parameter
            self.team = Team(
                name="Stock Analysis Team", 
                description="Team for comprehensive stock analysis",
                mode="coordinate",
                members=team_members,
                instructions=[
                    "You are a team of financial analysts working together to analyze stocks.",
                    "Each team member has a specific area of expertise and will contribute their analysis.",
                    "The coordinator will delegate tasks and synthesize the results into a comprehensive report."
                ],
                send_team_context_to_members=True,
                show_members_responses=True,
                markdown=True
                # DO NOT pass model parameter here - it's not supported in the current Agno version
            )
            self.logger.debug("Successfully created team")
        except Exception as e:
            self.logger.error(f"Error creating team: {e}")
            raise
        
        # Store the coordinator agent
        self.agent = coordinator
        
        # Add tools to the coordinator agent
        for tool in self.tools:
            self.agent.add_tool(tool)
            self.team.members.append(slave_agent.agent)
        
        self.logger.info(f"Team set up with {len(slave_agents)} slave agents in coordinate mode")
    
    def _delegate_analysis(self, 
                         analysis_type: str, 
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate analysis to a slave agent using our own coordination logic
        
        Args:
            analysis_type: Type of analysis to perform
            data: Data for the analysis
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Delegating {analysis_type} analysis using custom coordination logic")
        
        if not self.team:
            self.logger.error("Team not set up. Call setup_team() first.")
            return {
                "analysis_type": analysis_type,
                "status": "failed",
                "error": "Team not set up"
            }
        
        # Find appropriate slave agents for this analysis type
        appropriate_agents = []
        
        # Get all agent IDs in the team
        agent_ids = self.team.get_agent_ids()
        
        for agent_id in agent_ids:
            if agent_id == "coordinator":
                continue
                
            # Get the agent's role
            agent_info = self.team.get_agent_info(agent_id)
            if agent_info and 'role' in agent_info:
                # Check if the agent's role contains the analysis type
                if analysis_type.lower() in agent_info['role'].lower():
                    appropriate_agents.append(agent_id)
        
        if not appropriate_agents:
            self.logger.warning(f"No slave agents found for {analysis_type} analysis")
            return {
                "analysis_type": analysis_type,
                "status": "failed",
                "error": f"No slave agents available for {analysis_type} analysis"
            }
        
        # Create a task for the analysis
        task = {
            "analysis_type": analysis_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Implement our own coordination logic
        try:
            # For simplicity, we'll just use the first appropriate agent
            agent_id = appropriate_agents[0]
            self.logger.info(f"Selected agent {agent_id} for {analysis_type} analysis")
            
            # Get the agent instance from the team
            agent = self.team.get_agent(agent_id)
            if not agent:
                raise ValueError(f"Agent {agent_id} not found in team")
            
            # Create a prompt for the agent
            prompt = f"Perform {analysis_type} analysis on the provided data for ticker {data.get('ticker', 'unknown')}\n\nData: {json.dumps(data, indent=2)}"
            
            # Run the agent with the prompt
            result = agent.run(prompt)
            
            self.logger.info(f"Completed {analysis_type} analysis with agent {agent_id}")
            return {
                "analysis_type": analysis_type,
                "agent_id": agent_id,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Error in {analysis_type} analysis: {str(e)}")
            return {
                "analysis_type": analysis_type,
                "agent_ids": appropriate_agents,
                "status": "failed",
                "error": str(e)
            }
    
    def _aggregate_results_tool(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple analyses
        
        Args:
            results: List of analysis results
            
        Returns:
            Aggregated results
        """
        return self.aggregate_results({r.get("analysis_type", f"analysis_{i}"): r 
                                     for i, r in enumerate(results)})
    
    def aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple slave agents
        
        Args:
            results: Dictionary mapping analysis types to their results
            
        Returns:
            Aggregated results
        """
        self.logger.info(f"Aggregating results from {len(results)} analyses")
        
        # Extract key metrics and signals from each analysis
        aggregated = {
            "analyses": results,
            "completed_analyses": [k for k, v in results.items() 
                                  if v.get("status") == "completed"],
            "failed_analyses": [k for k, v in results.items() 
                               if v.get("status") == "failed"],
            "signals": {},
            "confidence_scores": {},
            "strengths": [],
            "weaknesses": [],
            "overall_recommendation": None
        }
        
        # Extract signals and confidence scores
        for analysis_type, result in results.items():
            if result.get("status") == "completed" and "result" in result:
                analysis_result = result["result"]
                
                # Extract signal if available
                if "signal" in analysis_result:
                    aggregated["signals"][analysis_type] = analysis_result["signal"]
                
                # Extract confidence score if available
                if "confidence_score" in analysis_result:
                    aggregated["confidence_scores"][analysis_type] = analysis_result["confidence_score"]
                
                # Extract strengths if available
                if "strengths" in analysis_result:
                    aggregated["strengths"].extend(analysis_result["strengths"])
                
                # Extract weaknesses if available
                if "weaknesses" in analysis_result:
                    aggregated["weaknesses"].extend(analysis_result["weaknesses"])
        
        # Calculate overall recommendation based on weighted signals
        if aggregated["signals"]:
            # This is a simplified example - in a real implementation,
            # you would use a more sophisticated algorithm
            signal_values = {
                "strong_buy": 2,
                "buy": 1,
                "hold": 0,
                "sell": -1,
                "strong_sell": -2
            }
            
            total_score = 0
            total_weight = 0
            
            for analysis_type, signal in aggregated["signals"].items():
                if signal.lower() in signal_values:
                    confidence = aggregated["confidence_scores"].get(analysis_type, 0.5)
                    weight = confidence
                    total_score += signal_values[signal.lower()] * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_score = total_score / total_weight
                
                # Convert score back to recommendation
                if avg_score >= 1.5:
                    aggregated["overall_recommendation"] = "Strong Buy"
                elif avg_score >= 0.5:
                    aggregated["overall_recommendation"] = "Buy"
                elif avg_score >= -0.5:
                    aggregated["overall_recommendation"] = "Hold"
                elif avg_score >= -1.5:
                    aggregated["overall_recommendation"] = "Sell"
                else:
                    aggregated["overall_recommendation"] = "Strong Sell"
        
        # Deduplicate strengths and weaknesses
        aggregated["strengths"] = list(set(aggregated["strengths"]))
        aggregated["weaknesses"] = list(set(aggregated["weaknesses"]))
        
        return aggregated
    
    def generate_final_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final report based on aggregated results
        
        Args:
            aggregated_results: Results aggregated from slave agents
            
        Returns:
            Final report
        """
        self.logger.info("Generating final report")
        
        # Prepare prompt for the agent
        prompt = self._prepare_final_report_prompt(aggregated_results)
        
        # Run the agent to generate the report
        try:
            response = self.agent.run(prompt)
            
            # Parse the response
            report = self._parse_final_report_response(response)
            
            # Add aggregated data to the report
            report["aggregated_data"] = aggregated_results
            
            self.logger.info("Final report generated successfully")
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating final report: {str(e)}")
            
            # Return a basic report with the error
            return {
                "error": str(e),
                "aggregated_data": aggregated_results,
                "report": "Error generating final report"
            }
    
    def _prepare_final_report_prompt(self, aggregated_results: Dict[str, Any]) -> str:
        """
        Prepare a prompt for generating the final report
        
        Args:
            aggregated_results: Aggregated results from all analyses
            
        Returns:
            Formatted prompt
        """
        prompt = f"""
        You are a financial analyst preparing a comprehensive stock analysis report.
        
        Here is the aggregated data from multiple specialized analyses:
        
        Overall Recommendation: {aggregated_results.get('overall_recommendation', 'N/A')}
        
        Completed Analyses: {', '.join(aggregated_results.get('completed_analyses', []))}
        
        Signals by Analysis Type:
        {json.dumps(aggregated_results.get('signals', {}), indent=2)}
        
        Confidence Scores:
        {json.dumps(aggregated_results.get('confidence_scores', {}), indent=2)}
        
        Key Strengths:
        {json.dumps(aggregated_results.get('strengths', []), indent=2)}
        
        Key Weaknesses:
        {json.dumps(aggregated_results.get('weaknesses', []), indent=2)}
        
        Based on this information, please generate a comprehensive stock analysis report with the following sections:
        
        1. Executive Summary
        2. Investment Recommendation
        3. Key Strengths
        4. Key Concerns
        5. Detailed Analysis by Category
        6. Valuation Assessment
        7. Risk Factors
        8. Conclusion
        
        Format your response as a well-structured report with clear headings for each section.
        """
        
        return prompt
    
    def _parse_final_report_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the final report response
        
        Args:
            response: Raw response from the agent
            
        Returns:
            Structured report
        """
        # For now, just return the raw response
        # In a more sophisticated implementation, you could parse the response
        # into a structured format with separate fields for each section
        
        return {
            "report": response
        }
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the stock analysis master agent using the Agno Teams coordination model
        
        Args:
            state: Dictionary containing input data, expects keys like:
                   'data': {'tickers': List[str], 'start_date': str, 'end_date': str}
                   'metadata': {'model_name': str, 'model_provider': str} (optional)
                   'slave_agents': Dictionary of slave agents to use
                   
        Returns:
            Dictionary containing the analysis results
        """
        self.logger.info("Starting stock analysis master agent with coordination model")
        self.progress.start()
        
        try:
            # Extract necessary info from state
            data = state.get("data", {})
            tickers = data.get("tickers", [])
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            
            # Get slave agents from state
            slave_agents = state.get("slave_agents", {})
            
            if not tickers:
                raise ValueError("No tickers provided for analysis")
            
            if not start_date or not end_date:
                raise ValueError("No date range provided for analysis")
            
            if not slave_agents:
                raise ValueError("No slave agents provided for analysis")
            
            # Set up the team with the provided slave agents
            self.setup_team(slave_agents)
            
            # Initialize results
            results = {}
            
            # Analyze each ticker
            for ticker in tickers:
                self.logger.info(f"Starting analysis for {ticker} with coordination model")
                
                # Update progress
                self.progress.update_status(f"data_prep_{ticker}", ProgressStatus.IN_PROGRESS, "Preparing data")
                
                # Get fundamental data
                fundamental_data = self._get_fundamental_data(ticker)
                
                # Get OHLCV data
                ohlcv_data = self._get_ohlcv_data(ticker)
                
                # Prepare data for analysis
                ticker_data = {
                    "ticker": ticker,
                    "fundamental_data": fundamental_data,
                    "ohlcv_data": ohlcv_data,
                    "start_date": start_date,
                    "end_date": end_date
                }
                
                # Mark data prep as complete
                self.progress.update_status(f"data_prep_{ticker}", ProgressStatus.COMPLETED, "Data preparation completed")
                
                # Create our own analysis plan for the ticker
                self.logger.info(f"Creating analysis plan for {ticker}")
                self.progress.update_status(f"planning_{ticker}", ProgressStatus.IN_PROGRESS, "Creating analysis plan")
                
                try:
                    # Define the analysis types we want to perform
                    analysis_types = self.analysis_types
                    
                    self.logger.info(f"Analysis plan created for {ticker}: {len(analysis_types)} analysis types")
                    self.progress.update_status(f"planning_{ticker}", ProgressStatus.COMPLETED, f"Plan created with {len(analysis_types)} analysis types")
                    
                    # Initialize analysis results for this ticker
                    ticker_results = {}
                    
                    # Execute each analysis type
                    for i, analysis_type in enumerate(analysis_types):
                        analysis_name = f"{analysis_type}_analysis"
                        self.progress.update_status(f"{analysis_name}_{ticker}", ProgressStatus.IN_PROGRESS, f"Executing {analysis_type} analysis")
                        
                        self.logger.info(f"Executing analysis {i+1}/{len(analysis_types)}: {analysis_type}")
                        
                        # Delegate the analysis to an appropriate agent
                        analysis_result = self._delegate_analysis(analysis_type, ticker_data)
                        
                        # Store the result
                        ticker_results[analysis_name] = {
                            "analysis_type": analysis_type,
                            "result": analysis_result,
                            "status": "completed" if analysis_result.get("status") == "completed" else "failed"
                        }
                        
                        self.progress.update_status(f"{analysis_name}_{ticker}", ProgressStatus.COMPLETED, f"Completed {analysis_type} analysis")
                    
                    # Synthesize final results
                    self.progress.update_status(f"synthesize_{ticker}", ProgressStatus.IN_PROGRESS, "Synthesizing results")
                    
                    # Implement our own synthesis logic
                    final_result = self._synthesize_results(ticker, ticker_results)
                    
                    # Store the result
                    results[ticker] = final_result
                    
                    self.progress.update_status(f"synthesize_{ticker}", ProgressStatus.COMPLETED, "Synthesis completed")
                    self.logger.info(f"Completed analysis for {ticker} with custom coordination logic")
                    
                except Exception as e:
                    self.logger.error(f"Error in coordination for {ticker}: {str(e)}")
                    results[ticker] = {
                        "error": str(e),
                        "status": "failed"
                    }
            
            # Mark as completed
            self.progress.complete(message="Stock analysis completed successfully")
            
            return {"results": results}
            
        except Exception as e:
            self.logger.error(f"Error in stock analysis: {str(e)}")
            self.progress.fail(message=f"Error in stock analysis: {str(e)}")
            
            # Return error
            return {"error": str(e)}
    
    def _synthesize_results(self, ticker: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize the results from all analyses into a final recommendation
        
        Args:
            ticker: Stock ticker
            analysis_results: Results from all analyses
            
        Returns:
            Synthesized result with final recommendation
        """
        self.logger.info(f"Synthesizing results for {ticker} from {len(analysis_results)} analyses")
        
        try:
            # Extract signals and confidence scores from each analysis
            signals = []
            confidence_scores = []
            strengths = []
            weaknesses = []
            metrics = {}
            reasoning_points = []
            
            # Process each analysis result
            for analysis_name, analysis_data in analysis_results.items():
                if analysis_data.get("status") != "completed":
                    continue
                    
                result = analysis_data.get("result", {})
                
                # Extract the signal and confidence
                if isinstance(result, dict):
                    # Get signal
                    signal = result.get("signal")
                    if signal:
                        signals.append(signal)
                    
                    # Get confidence
                    confidence = result.get("confidence")
                    if confidence is not None and isinstance(confidence, (int, float)):
                        confidence_scores.append(confidence)
                    
                    # Get strengths
                    if "strengths" in result and isinstance(result["strengths"], list):
                        strengths.extend(result["strengths"])
                    
                    # Get weaknesses
                    if "weaknesses" in result and isinstance(result["weaknesses"], list):
                        weaknesses.extend(result["weaknesses"])
                    
                    # Get metrics
                    if "metrics" in result and isinstance(result["metrics"], dict):
                        metrics.update(result["metrics"])
                    
                    # Get reasoning
                    reasoning = result.get("reasoning")
                    if reasoning:
                        reasoning_points.append(f"{analysis_data.get('analysis_type', 'Analysis')}: {reasoning}")
            
            # Determine the overall signal
            overall_signal = "neutral"
            if signals:
                bullish_count = sum(1 for s in signals if s in ["bullish", "buy", "strong buy"])
                bearish_count = sum(1 for s in signals if s in ["bearish", "sell", "strong sell"])
                
                if bullish_count > bearish_count:
                    overall_signal = "bullish"
                elif bearish_count > bullish_count:
                    overall_signal = "bearish"
            
            # Calculate the overall confidence
            overall_confidence = 0.5  # Default
            if confidence_scores:
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Prepare the final reasoning
            overall_reasoning = "\n\n".join(reasoning_points)
            
            # Create the final result
            final_result = {
                "ticker": ticker,
                "signal": overall_signal,
                "confidence": overall_confidence,
                "strengths": strengths[:5],  # Limit to top 5 strengths
                "weaknesses": weaknesses[:5],  # Limit to top 5 weaknesses
                "metrics": metrics,
                "reasoning": overall_reasoning,
                "analysis_by_type": analysis_results,
                "status": "completed"
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing results for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "signal": "neutral",
                "confidence": 0.5,
                "reasoning": f"Error synthesizing results: {str(e)}",
                "status": "failed",
                "error": str(e)
            }
