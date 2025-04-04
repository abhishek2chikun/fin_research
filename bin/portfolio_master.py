"""
Portfolio Management Master Agent

This module implements a master agent that orchestrates multiple slave agents 
for comprehensive portfolio management using Agno Teams.
"""

from typing import Dict, List, Optional, Union, Any, Callable
import logging
from datetime import datetime
import json

# Import Agno team components
from bin.base_agent import MasterAgent
from bin.progress_util import ProgressTracker, ProgressStatus
from bin.agno_wrapper import Tool

# Import data models
from data.fundamental_data import FinancialData
from data.ohlcv import OHLCVData


class PortfolioMaster(MasterAgent):
    """
    Master agent for orchestrating portfolio management
    
    This agent coordinates multiple slave agents to perform various portfolio
    management tasks, including position sizing, risk management, and rebalancing.
    """
    
    def __init__(self, 
               model_name: str = "gpt-4", 
               model_provider: str = "openai"):
        """
        Initialize the portfolio management master agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
        """
        super().__init__(model_name, model_provider)
        
        # Portfolio management stages
        self.stages = [
            "portfolio_analysis",
            "risk_assessment",
            "position_sizing",
            "rebalancing",
            "optimization"
        ]
        
        # Initialize progress tracker
        self.progress = ProgressTracker(
            total_steps=len(self.stages) + 2,  # Stages + data prep + final report
            name="PortfolioManagement"
        )
        
        # Set up tools
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools for the agent"""
        
        # Tool for getting portfolio data
        get_portfolio_tool = Tool(
            name="get_portfolio_data",
            func=self._get_portfolio_data,
            description="Get current portfolio data"
        )
        
        # Tool for delegating to risk manager
        risk_assessment_tool = Tool(
            name="assess_risk",
            func=self._assess_risk,
            description="Assess portfolio risk using risk manager agent"
        )
        
        # Tool for delegating to position sizing agent
        position_sizing_tool = Tool(
            name="calculate_position_sizes",
            func=self._calculate_position_sizes,
            description="Calculate position sizes using position sizing agent"
        )
        
        # Tool for portfolio rebalancing
        rebalance_portfolio_tool = Tool(
            name="rebalance_portfolio",
            func=self._rebalance_portfolio,
            description="Generate portfolio rebalancing recommendations"
        )
        
        # Tool for portfolio optimization
        optimize_portfolio_tool = Tool(
            name="optimize_portfolio",
            func=self._optimize_portfolio,
            description="Optimize portfolio for risk/return"
        )
        
        # Add tools to the agent
        self.add_tools([
            get_portfolio_tool,
            risk_assessment_tool,
            position_sizing_tool,
            rebalance_portfolio_tool,
            optimize_portfolio_tool
        ])
    
    def _get_portfolio_data(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get current portfolio data
        
        Args:
            user_id: Optional user ID to get portfolio for
            
        Returns:
            Portfolio data
        """
        self.logger.info(f"Getting portfolio data for user {user_id or 'default'}")
        
        # Implement the logic to get portfolio data
        # This is a placeholder - in a real implementation, you would fetch
        # the actual portfolio from a database or other source
        
        # Example portfolio data
        portfolio = {
            "total_value": 1000000,
            "cash": 100000,
            "positions": [
                {"ticker": "AAPL", "shares": 100, "cost_basis": 150, "current_price": 175, "weight": 0.15},
                {"ticker": "MSFT", "shares": 80, "cost_basis": 280, "current_price": 310, "weight": 0.20},
                {"ticker": "GOOGL", "shares": 50, "cost_basis": 110, "current_price": 130, "weight": 0.10},
                {"ticker": "AMZN", "shares": 30, "cost_basis": 3300, "current_price": 3500, "weight": 0.15},
                {"ticker": "BRK.B", "shares": 120, "cost_basis": 270, "current_price": 285, "weight": 0.12},
                {"ticker": "JNJ", "shares": 200, "cost_basis": 160, "current_price": 165, "weight": 0.08},
                {"ticker": "JPM", "shares": 150, "cost_basis": 140, "current_price": 155, "weight": 0.10}
            ],
            "target_allocation": {
                "Technology": 0.40,
                "Healthcare": 0.15,
                "Financials": 0.15,
                "Consumer": 0.15,
                "Other": 0.15
            },
            "current_allocation": {
                "Technology": 0.45,
                "Healthcare": 0.08,
                "Financials": 0.10,
                "Consumer": 0.15,
                "Other": 0.12,
                "Cash": 0.10
            }
        }
        
        return portfolio
    
    def _assess_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess portfolio risk using risk manager agent
        
        Args:
            portfolio_data: Current portfolio data
            
        Returns:
            Risk assessment
        """
        self.logger.info("Delegating risk assessment to risk manager agent")
        
        # Find the risk manager agent
        risk_manager = None
        for agent_id, info in self.slave_agents.items():
            agent_instance = info["instance"]
            if hasattr(agent_instance, 'analysis_type') and agent_instance.analysis_type == "risk":
                risk_manager = agent_id
                break
        
        if not risk_manager:
            self.logger.warning("No risk manager agent found")
            return {
                "status": "failed",
                "error": "No risk manager agent available"
            }
        
        # Create task for the risk manager
        task = {
            "type": "risk_assessment",
            "portfolio_data": portfolio_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Delegate to the risk manager
        try:
            result = self.delegate_task(risk_manager, task)
            self.logger.info("Completed risk assessment")
            return {
                "status": "completed",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _calculate_position_sizes(self, 
                              portfolio_data: Dict[str, Any],
                              risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate position sizes using position sizing agent
        
        Args:
            portfolio_data: Current portfolio data
            risk_assessment: Risk assessment data
            
        Returns:
            Position sizing recommendations
        """
        self.logger.info("Delegating position sizing to position sizing agent")
        
        # Find the position sizing agent
        position_sizer = None
        for agent_id, info in self.slave_agents.items():
            agent_instance = info["instance"]
            if hasattr(agent_instance, 'analysis_type') and agent_instance.analysis_type == "position_sizing":
                position_sizer = agent_id
                break
        
        if not position_sizer:
            self.logger.warning("No position sizing agent found")
            return {
                "status": "failed",
                "error": "No position sizing agent available"
            }
        
        # Create task for the position sizer
        task = {
            "type": "position_sizing",
            "portfolio_data": portfolio_data,
            "risk_assessment": risk_assessment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Delegate to the position sizer
        try:
            result = self.delegate_task(position_sizer, task)
            self.logger.info("Completed position sizing")
            return {
                "status": "completed",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Error in position sizing: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _rebalance_portfolio(self, 
                         portfolio_data: Dict[str, Any],
                         position_sizes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate portfolio rebalancing recommendations
        
        Args:
            portfolio_data: Current portfolio data
            position_sizes: Position sizing recommendations
            
        Returns:
            Rebalancing recommendations
        """
        self.logger.info("Generating portfolio rebalancing recommendations")
        
        # Current allocation
        current_allocation = portfolio_data.get("current_allocation", {})
        
        # Target allocation
        target_allocation = portfolio_data.get("target_allocation", {})
        
        # Calculate differences
        differences = {}
        for category, target in target_allocation.items():
            current = current_allocation.get(category, 0)
            differences[category] = {
                "current": current,
                "target": target,
                "difference": target - current
            }
        
        # Generate rebalancing actions
        actions = []
        for category, diff in differences.items():
            if abs(diff["difference"]) >= 0.02:  # 2% threshold for rebalancing
                if diff["difference"] > 0:
                    actions.append({
                        "action": "increase",
                        "category": category,
                        "amount": diff["difference"] * portfolio_data["total_value"],
                        "percentage": diff["difference"] * 100
                    })
                else:
                    actions.append({
                        "action": "decrease",
                        "category": category,
                        "amount": -diff["difference"] * portfolio_data["total_value"],
                        "percentage": -diff["difference"] * 100
                    })
        
        # Generate specific trade recommendations
        # This would normally use more sophisticated logic, but for simplicity,
        # we'll just generate placeholder recommendations
        trades = []
        for action in actions:
            category = action["category"]
            positions = [p for p in portfolio_data["positions"] 
                        if self._get_position_category(p["ticker"]) == category]
            
            if action["action"] == "increase" and positions:
                # Buy more of existing positions in this category
                for position in positions:
                    trades.append({
                        "action": "buy",
                        "ticker": position["ticker"],
                        "amount": action["amount"] / len(positions),
                        "reasoning": f"Increase allocation to {category}"
                    })
            elif action["action"] == "decrease" and positions:
                # Sell some of existing positions in this category
                for position in positions:
                    trades.append({
                        "action": "sell",
                        "ticker": position["ticker"],
                        "amount": action["amount"] / len(positions),
                        "reasoning": f"Decrease allocation to {category}"
                    })
        
        return {
            "status": "completed",
            "differences": differences,
            "actions": actions,
            "trades": trades
        }
    
    def _get_position_category(self, ticker: str) -> str:
        """
        Get the category for a position
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Category name
        """
        # This is a placeholder - in a real implementation, you would use
        # a more sophisticated method to categorize stocks
        
        categories = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer",
            "BRK.B": "Financials",
            "JNJ": "Healthcare",
            "JPM": "Financials"
        }
        
        return categories.get(ticker, "Other")
    
    def _optimize_portfolio(self, 
                         portfolio_data: Dict[str, Any],
                         risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio for risk/return
        
        Args:
            portfolio_data: Current portfolio data
            risk_assessment: Risk assessment data
            
        Returns:
            Optimization recommendations
        """
        self.logger.info("Optimizing portfolio for risk/return")
        
        # This would normally involve more sophisticated optimization algorithms,
        # but for simplicity, we'll just generate placeholder recommendations
        
        # Extract current positions
        positions = portfolio_data.get("positions", [])
        
        # Calculate simple metrics
        total_value = portfolio_data.get("total_value", 0)
        position_count = len(positions)
        
        # Generate optimization recommendations
        recommendations = []
        
        # Check for over-concentration
        for position in positions:
            if position.get("weight", 0) > 0.15:
                recommendations.append({
                    "type": "reduce_concentration",
                    "ticker": position["ticker"],
                    "current_weight": position["weight"],
                    "target_weight": 0.15,
                    "reasoning": f"{position['ticker']} represents {position['weight']*100:.1f}% of the portfolio, which exceeds the 15% concentration limit."
                })
        
        # Check for under-diversification
        if position_count < 10:
            recommendations.append({
                "type": "increase_diversification",
                "current_positions": position_count,
                "target_positions": 10,
                "reasoning": f"Portfolio contains only {position_count} positions, which may not provide adequate diversification."
            })
        
        # Check cash allocation
        cash_allocation = portfolio_data.get("current_allocation", {}).get("Cash", 0)
        if cash_allocation < 0.05:
            recommendations.append({
                "type": "increase_cash",
                "current_cash": cash_allocation,
                "target_cash": 0.05,
                "reasoning": f"Cash allocation of {cash_allocation*100:.1f}% is below the recommended 5% minimum for liquidity."
            })
        elif cash_allocation > 0.15:
            recommendations.append({
                "type": "decrease_cash",
                "current_cash": cash_allocation,
                "target_cash": 0.15,
                "reasoning": f"Cash allocation of {cash_allocation*100:.1f}% is above the recommended 15% maximum and may drag on returns."
            })
        
        return {
            "status": "completed",
            "recommendations": recommendations
        }
    
    def aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple portfolio management analyses
        
        Args:
            results: Dictionary mapping analysis types to their results
            
        Returns:
            Aggregated results
        """
        self.logger.info(f"Aggregating results from {len(results)} portfolio analyses")
        
        # Extract key components from each analysis
        aggregated = {
            "analyses": results,
            "completed_analyses": [k for k, v in results.items() 
                                  if v.get("status") == "completed"],
            "failed_analyses": [k for k, v in results.items() 
                               if v.get("status") == "failed"],
            "risk_assessment": results.get("risk_assessment", {}).get("result", {}),
            "position_sizing": results.get("position_sizing", {}).get("result", {}),
            "rebalancing": results.get("rebalancing", {}).get("result", {}),
            "optimization": results.get("optimization", {}).get("result", {})
        }
        
        # Combine recommendations from different analyses
        all_recommendations = []
        
        # Add rebalancing trades
        if "rebalancing" in results and results["rebalancing"].get("status") == "completed":
            rebalancing = results["rebalancing"]["result"]
            if "trades" in rebalancing:
                for trade in rebalancing["trades"]:
                    all_recommendations.append({
                        "type": "rebalancing",
                        "action": trade["action"],
                        "ticker": trade["ticker"],
                        "amount": trade["amount"],
                        "reasoning": trade["reasoning"]
                    })
        
        # Add optimization recommendations
        if "optimization" in results and results["optimization"].get("status") == "completed":
            optimization = results["optimization"]["result"]
            if "recommendations" in optimization:
                for rec in optimization["recommendations"]:
                    all_recommendations.append({
                        "type": "optimization",
                        "action": rec["type"],
                        "details": rec,
                        "reasoning": rec["reasoning"]
                    })
        
        # Combine position sizing recommendations
        if "position_sizing" in results and results["position_sizing"].get("status") == "completed":
            position_sizing = results["position_sizing"]["result"]
            if "position_sizes" in position_sizing:
                for ticker, size in position_sizing["position_sizes"].items():
                    all_recommendations.append({
                        "type": "position_sizing",
                        "action": "adjust_position",
                        "ticker": ticker,
                        "target_size": size,
                        "reasoning": f"Position size adjusted based on risk parameters"
                    })
        
        # Add combined recommendations to aggregated results
        aggregated["all_recommendations"] = all_recommendations
        
        return aggregated
    
    def generate_final_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final report based on aggregated results
        
        Args:
            aggregated_results: Results aggregated from portfolio analyses
            
        Returns:
            Final report
        """
        self.logger.info("Generating final portfolio management report")
        
        # Prepare prompt for the agent
        prompt = self._prepare_final_report_prompt(aggregated_results)
        
        # Run the agent to generate the report
        try:
            response = self.agent.run(prompt)
            
            # Parse the response
            report = self._parse_final_report_response(response)
            
            # Add aggregated data to the report
            report["aggregated_data"] = aggregated_results
            
            self.logger.info("Final portfolio management report generated successfully")
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating final portfolio management report: {str(e)}")
            
            # Return a basic report with the error
            return {
                "error": str(e),
                "aggregated_data": aggregated_results,
                "report": "Error generating final portfolio management report"
            }
    
    def _prepare_final_report_prompt(self, aggregated_results: Dict[str, Any]) -> str:
        """
        Prepare a prompt for generating the final report
        
        Args:
            aggregated_results: Aggregated results from all portfolio analyses
            
        Returns:
            Formatted prompt
        """
        all_recommendations = aggregated_results.get("all_recommendations", [])
        risk_assessment = aggregated_results.get("risk_assessment", {})
        
        prompt = f"""
        You are a portfolio manager preparing a comprehensive portfolio management report.
        
        Here is the aggregated data from multiple portfolio analyses:
        
        Risk Assessment:
        {json.dumps(risk_assessment, indent=2)}
        
        Number of Recommendations: {len(all_recommendations)}
        
        Recommendations:
        {json.dumps(all_recommendations[:10], indent=2)}  # Show first 10 for brevity
        
        Based on this information, please generate a comprehensive portfolio management report with the following sections:
        
        1. Executive Summary
        2. Portfolio Overview
        3. Risk Assessment
        4. Rebalancing Recommendations
        5. Position Sizing Recommendations
        6. Optimization Recommendations
        7. Implementation Strategy
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
        Run the portfolio management master agent
        
        Args:
            state: Current state of the analysis
            
        Returns:
            Updated state with portfolio management results
        """
        self.logger.info("Starting portfolio management master agent")
        self.progress.start()
        
        try:
            # Step 1: Get portfolio data
            self.progress.update(1, message="Getting portfolio data")
            
            # Get user ID from state or use default
            user_id = state.get("user_id")
            
            # Get portfolio data
            portfolio_data = self._get_portfolio_data(user_id)
            
            # Update state
            state["portfolio_data"] = portfolio_data
            
            # Initialize results dictionary
            results = {}
            
            # Step 2: Assess portfolio risk
            self.progress.update(2, message="Assessing portfolio risk")
            
            risk_assessment = self._assess_risk(portfolio_data)
            results["risk_assessment"] = risk_assessment
            
            # Step 3: Calculate position sizes
            self.progress.update(3, message="Calculating position sizes")
            
            position_sizing = self._calculate_position_sizes(
                portfolio_data,
                risk_assessment.get("result", {})
            )
            results["position_sizing"] = position_sizing
            
            # Step 4: Generate rebalancing recommendations
            self.progress.update(4, message="Generating rebalancing recommendations")
            
            rebalancing = self._rebalance_portfolio(
                portfolio_data,
                position_sizing.get("result", {})
            )
            results["rebalancing"] = rebalancing
            
            # Step 5: Optimize portfolio
            self.progress.update(5, message="Optimizing portfolio")
            
            optimization = self._optimize_portfolio(
                portfolio_data,
                risk_assessment.get("result", {})
            )
            results["optimization"] = optimization
            
            # Step 6: Aggregate results
            self.progress.update(6, message="Aggregating results")
            
            aggregated_results = self.aggregate_results(results)
            
            # Step 7: Generate final report
            self.progress.update(7, message="Generating final report")
            
            final_report = self.generate_final_report(aggregated_results)
            
            # Update state with results
            state["portfolio_results"] = results
            state["aggregated_results"] = aggregated_results
            state["final_report"] = final_report
            
            # Mark as completed
            self.progress.complete(message="Portfolio management completed successfully")
            
            return state
        except Exception as e:
            self.logger.error(f"Error in portfolio management: {str(e)}")
            self.progress.fail(message=f"Error in portfolio management: {str(e)}")
            
            # Update state with error
            state["error"] = str(e)
            
            return state
