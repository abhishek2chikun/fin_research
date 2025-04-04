"""
Sector Analysis Master Agent

This module implements a master agent that orchestrates multiple slave agents 
for comprehensive sector analysis using Agno Teams.
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


class SectorAnalysisMaster(MasterAgent):
    """
    Master agent for orchestrating sector analysis
    
    This agent coordinates multiple slave agents to perform comprehensive
    sector analysis and synthesizes their results into a final report with
    comparative rankings and recommendations.
    """
    
    def __init__(self, 
               model_name: str = "gpt-4", 
               model_provider: str = "openai",
               analysis_types: List[str] = None):
        """
        Initialize the sector analysis master agent
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM
            analysis_types: Types of analysis to perform
        """
        super().__init__(model_name, model_provider)
        
        # Set up analysis types
        self.analysis_types = analysis_types or [
            "value", "growth", "quality", "technical", "sentiment", "valuation"
        ]
        
        # Sector ETF mapping
        self.sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Materials": "XLB",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC"
        }
        
        # Initialize progress tracker
        self.progress = ProgressTracker(
            total_steps=4,  # Data prep, stock analysis, comparison, final report
            name="SectorAnalysis"
        )
        
        # Set up tools
        self._setup_tools()
    
    def _setup_tools(self):
        """Set up tools for the agent"""
        
        # Tool for getting sector stocks
        get_sector_stocks_tool = Tool(
            name="get_sector_stocks",
            func=self._get_sector_stocks,
            description="Get stocks in a specific sector"
        )
        
        # Tool for analyzing individual stocks
        analyze_stock_tool = Tool(
            name="analyze_stock",
            func=self._analyze_stock,
            description="Analyze an individual stock using slave agents"
        )
        
        # Tool for comparing stocks
        compare_stocks_tool = Tool(
            name="compare_stocks",
            func=self._compare_stocks,
            description="Compare multiple stocks within a sector"
        )
        
        # Tool for getting sector ETF data
        get_sector_etf_tool = Tool(
            name="get_sector_etf",
            func=self._get_sector_etf,
            description="Get data for a sector ETF"
        )
        
        # Add tools to the agent
        self.add_tools([
            get_sector_stocks_tool,
            analyze_stock_tool,
            compare_stocks_tool,
            get_sector_etf_tool
        ])
    
    def _get_sector_stocks(self, sector: str, limit: int = 10) -> List[str]:
        """
        Get stocks in a specific sector
        
        Args:
            sector: Sector name
            limit: Maximum number of stocks to return
            
        Returns:
            List of stock tickers in the sector
        """
        self.logger.info(f"Getting stocks in {sector} sector (limit: {limit})")
        
        # Implement the logic to get stocks in the sector
        # This is a placeholder - in a real implementation, you would use
        # a data provider or load from a database
        
        # For now, return placeholder data
        sector_stocks = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "CSCO", "ORCL", "ADBE", "CRM", "AMD"],
            "Healthcare": ["JNJ", "UNH", "PFE", "ABT", "MRK", "TMO", "ABBV", "DHR", "LLY", "AMGN"],
            "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "EBAY"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "EL", "CL", "GIS"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "KMI"],
            "Materials": ["LIN", "APD", "ECL", "DD", "NEM", "FCX", "NUE", "DOW", "PPG", "SHW"],
            "Industrials": ["HON", "UNP", "UPS", "BA", "RTX", "CAT", "DE", "GE", "LMT", "MMM"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "ED"],
            "Real Estate": ["AMT", "PLD", "CCI", "PSA", "EQIX", "DLR", "O", "SPG", "WELL", "AVB"],
            "Communication Services": ["GOOGL", "META", "NFLX", "CMCSA", "VZ", "T", "TMUS", "ATVI", "EA", "CHTR"]
        }
        
        # Return stocks for the requested sector (or empty list if sector not found)
        stocks = sector_stocks.get(sector, [])
        
        # Limit the number of stocks
        return stocks[:limit]
    
    def _get_sector_etf(self, sector: str) -> Dict[str, Any]:
        """
        Get data for a sector ETF
        
        Args:
            sector: Sector name
            
        Returns:
            ETF data
        """
        etf_ticker = self.sector_etfs.get(sector)
        
        if not etf_ticker:
            self.logger.warning(f"No ETF found for sector {sector}")
            return {"sector": sector, "etf_ticker": None, "data_available": False}
        
        self.logger.info(f"Getting data for sector ETF {etf_ticker} ({sector})")
        
        # Implement the logic to get ETF data
        # This is a placeholder - in a real implementation, you would use
        # a data provider or load from a database
        
        # For now, return placeholder data
        return {
            "sector": sector,
            "etf_ticker": etf_ticker,
            "data_available": True
        }
    
    def _analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze an individual stock using slave agents
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing stock {ticker}")
        
        # Create a state dictionary for the stock
        stock_state = {"ticker": ticker}
        
        # Find a stock analysis master agent
        stock_master = None
        for agent_id, info in self.slave_agents.items():
            agent_instance = info["instance"]
            
            # Check if this is a StockAnalysisMaster
            if agent_instance.__class__.__name__ == "StockAnalysisMaster":
                stock_master = agent_instance
                break
        
        if not stock_master:
            self.logger.warning(f"No StockAnalysisMaster agent found for analyzing {ticker}")
            return {
                "ticker": ticker,
                "status": "failed",
                "error": "No StockAnalysisMaster agent available"
            }
        
        # Delegate the analysis to the stock master agent
        try:
            result = stock_master.run(stock_state)
            self.logger.info(f"Completed analysis of {ticker}")
            return {
                "ticker": ticker,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "status": "failed",
                "error": str(e)
            }
    
    def _compare_stocks(self, 
                      stock_analyses: Dict[str, Dict[str, Any]],
                      metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple stocks within a sector
        
        Args:
            stock_analyses: Dictionary mapping tickers to their analysis results
            metrics: List of metrics to compare
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(stock_analyses)} stocks")
        
        # Default metrics if none provided
        if not metrics:
            metrics = [
                "value", "growth", "quality", "technical", "sentiment", "valuation"
            ]
        
        # Extract relevant metrics from each stock's analysis
        comparison = {
            "stocks": list(stock_analyses.keys()),
            "metrics": metrics,
            "rankings": {},
            "overall_ranking": [],
            "top_picks": []
        }
        
        # Rank stocks by each metric
        for metric in metrics:
            metric_scores = {}
            
            for ticker, analysis in stock_analyses.items():
                if analysis.get("status") != "completed" or "result" not in analysis:
                    continue
                
                result = analysis["result"]
                
                # Extract the aggregated results
                if "aggregated_results" not in result:
                    continue
                
                aggregated = result["aggregated_results"]
                
                # Extract signals and confidence scores
                signals = aggregated.get("signals", {})
                confidence_scores = aggregated.get("confidence_scores", {})
                
                # Assign a score based on the signal for this metric
                if metric in signals:
                    signal = signals[metric]
                    confidence = confidence_scores.get(metric, 0.5)
                    
                    # Convert signal to numeric score
                    signal_values = {
                        "strong_buy": 5,
                        "buy": 4,
                        "hold": 3,
                        "sell": 2,
                        "strong_sell": 1
                    }
                    
                    # Get numeric score or default to middle value
                    score = signal_values.get(signal.lower(), 3)
                    
                    # Apply confidence weighting
                    weighted_score = score * confidence
                    
                    metric_scores[ticker] = weighted_score
            
            # Rank stocks by this metric
            ranked_stocks = sorted(
                metric_scores.keys(),
                key=lambda ticker: metric_scores[ticker],
                reverse=True  # Higher scores first
            )
            
            comparison["rankings"][metric] = ranked_stocks
        
        # Calculate overall ranking based on average rank across metrics
        overall_scores = {}
        
        for ticker in comparison["stocks"]:
            overall_scores[ticker] = 0
            metric_count = 0
            
            for metric in metrics:
                if metric in comparison["rankings"]:
                    ranking = comparison["rankings"][metric]
                    
                    if ticker in ranking:
                        # Get position in ranking (0-indexed)
                        position = ranking.index(ticker)
                        
                        # Convert to score (higher is better)
                        score = len(ranking) - position
                        
                        overall_scores[ticker] += score
                        metric_count += 1
            
            # Calculate average score
            if metric_count > 0:
                overall_scores[ticker] = overall_scores[ticker] / metric_count
        
        # Sort by overall score
        comparison["overall_ranking"] = sorted(
            overall_scores.keys(),
            key=lambda ticker: overall_scores[ticker],
            reverse=True  # Higher scores first
        )
        
        # Select top picks (top 3 or fewer)
        comparison["top_picks"] = comparison["overall_ranking"][:min(3, len(comparison["overall_ranking"]))]
        
        return comparison
    
    def aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple stock analyses
        
        Args:
            results: Dictionary mapping tickers to their analysis results
            
        Returns:
            Aggregated results
        """
        self.logger.info(f"Aggregating results from {len(results)} stock analyses")
        
        # Compare stocks
        comparison = self._compare_stocks(results)
        
        # Extract sector information
        sector = None
        for ticker, analysis in results.items():
            if analysis.get("status") == "completed" and "result" in analysis:
                if "sector" in analysis["result"]:
                    sector = analysis["result"]["sector"]
                    break
        
        # Create aggregated result
        aggregated = {
            "sector": sector,
            "analyzed_stocks": list(results.keys()),
            "comparison": comparison,
            "top_picks": comparison["top_picks"],
            "stocks_by_metric": comparison["rankings"],
            "overall_ranking": comparison["overall_ranking"]
        }
        
        return aggregated
    
    def generate_final_report(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a final report based on aggregated results
        
        Args:
            aggregated_results: Results aggregated from stock analyses
            
        Returns:
            Final report
        """
        self.logger.info("Generating final sector analysis report")
        
        # Prepare prompt for the agent
        prompt = self._prepare_final_report_prompt(aggregated_results)
        
        # Run the agent to generate the report
        try:
            response = self.agent.run(prompt)
            
            # Parse the response
            report = self._parse_final_report_response(response)
            
            # Add aggregated data to the report
            report["aggregated_data"] = aggregated_results
            
            self.logger.info("Final sector analysis report generated successfully")
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating final sector analysis report: {str(e)}")
            
            # Return a basic report with the error
            return {
                "error": str(e),
                "aggregated_data": aggregated_results,
                "report": "Error generating final sector analysis report"
            }
    
    def _prepare_final_report_prompt(self, aggregated_results: Dict[str, Any]) -> str:
        """
        Prepare a prompt for generating the final report
        
        Args:
            aggregated_results: Aggregated results from all stock analyses
            
        Returns:
            Formatted prompt
        """
        sector = aggregated_results.get("sector", "Unknown")
        analyzed_stocks = aggregated_results.get("analyzed_stocks", [])
        top_picks = aggregated_results.get("top_picks", [])
        overall_ranking = aggregated_results.get("overall_ranking", [])
        
        prompt = f"""
        You are a financial analyst preparing a comprehensive sector analysis report.
        
        Sector: {sector}
        
        Analyzed Stocks: {', '.join(analyzed_stocks)}
        
        Top Picks: {', '.join(top_picks)}
        
        Overall Ranking:
        {json.dumps(overall_ranking, indent=2)}
        
        Rankings by Metric:
        {json.dumps(aggregated_results.get('stocks_by_metric', {}), indent=2)}
        
        Based on this information, please generate a comprehensive sector analysis report with the following sections:
        
        1. Executive Summary
        2. Sector Overview and Trends
        3. Top Stock Picks with Rationale
        4. Comparative Analysis
        5. Performance by Metric
        6. Investment Opportunities and Risks
        7. Sector Outlook
        8. Conclusion and Recommendations
        
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
        Run the sector analysis master agent
        
        Args:
            state: Current state of the analysis
            
        Returns:
            Updated state with analysis results
        """
        self.logger.info("Starting sector analysis master agent")
        self.progress.start()
        
        try:
            # Extract sector from state
            sector = state.get("sector")
            if not sector:
                raise ValueError("Sector not provided in state")
            
            # Step 1: Get stocks in the sector
            self.progress.update(1, message=f"Getting stocks in {sector} sector")
            
            # Get limit from state or use default
            limit = state.get("limit", 5)
            
            # Get stocks in the sector
            sector_stocks = self._get_sector_stocks(sector, limit)
            
            if not sector_stocks:
                raise ValueError(f"No stocks found in {sector} sector")
            
            # Get sector ETF data
            sector_etf = self._get_sector_etf(sector)
            
            # Update state
            state["sector_stocks"] = sector_stocks
            state["sector_etf"] = sector_etf
            
            # Step 2: Analyze each stock
            self.progress.update(2, message=f"Analyzing {len(sector_stocks)} stocks")
            
            stock_analyses = {}
            
            for ticker in sector_stocks:
                self.logger.info(f"Analyzing stock {ticker} in {sector} sector")
                
                # Analyze the stock
                analysis = self._analyze_stock(ticker)
                
                # Add sector information
                if "result" in analysis:
                    analysis["result"]["sector"] = sector
                
                # Store the analysis
                stock_analyses[ticker] = analysis
            
            # Update state
            state["stock_analyses"] = stock_analyses
            
            # Step 3: Compare stocks
            self.progress.update(3, message="Comparing stocks")
            
            # Aggregate results
            aggregated_results = self.aggregate_results(stock_analyses)
            
            # Update state
            state["aggregated_results"] = aggregated_results
            
            # Step 4: Generate final report
            self.progress.update(4, message="Generating final report")
            
            # Generate final report
            final_report = self.generate_final_report(aggregated_results)
            
            # Update state
            state["final_report"] = final_report
            
            # Mark as completed
            self.progress.complete(message="Sector analysis completed successfully")
            
            return state
        except Exception as e:
            self.logger.error(f"Error in sector analysis: {str(e)}")
            self.progress.fail(message=f"Error in sector analysis: {str(e)}")
            
            # Update state with error
            state["error"] = str(e)
            
            return state
