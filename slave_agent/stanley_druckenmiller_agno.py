"""
Stanley Druckenmiller Analysis Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
import statistics
import re
from pydantic import BaseModel
from typing_extensions import Literal
import pandas as pd
import os

# Import Agno framework
from agno.agent import Agent
from agno.models.lmstudio import LMStudio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools
from tools.ohlcv import fetch_ohlcv_data, OHLCVData, convert_df_to_price_data, load_filtered_price_data
from tools.fundamental import fetch_fundamental_data, get_market_cap
from tools.fundamental import get_pe_ratio, get_free_cash_flow

# Pydantic model for the output signal
class DruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def analyze_growth_and_momentum(ticker: str, prices_data = None) -> Dict[str, Any]:
    """
    Evaluate:
      - Revenue Growth (YoY)
      - EPS Growth (YoY)
      - Price Momentum
    
    Returns a dictionary with score and detailed analysis.
    """
    # Fetch fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    
    details = []
    raw_score = 0  # We'll sum up a maximum of 9 raw points, then scale to 0–10
    
    # 1. Revenue Growth
    revenue_found = False
    for revenue_key in ["Revenue", "Sales", "Sales\xa0+", "Total Revenue", "Net Sales"]:
        if fundamental_data and "profit_loss" in fundamental_data and revenue_key in fundamental_data["profit_loss"]:
            revenues = fundamental_data["profit_loss"][revenue_key]
            revenues = [float(r) if r is not None else None for r in revenues]
            valid_revenues = [r for r in revenues if r is not None]
            revenue_found = True
            
            if len(valid_revenues) >= 2:
                latest_rev = valid_revenues[0]
                older_rev = valid_revenues[-1]
                if older_rev > 0:
                    rev_growth = (latest_rev - older_rev) / abs(older_rev)
                    if rev_growth > 0.30:
                        raw_score += 3
                        details.append(f"Strong revenue growth: {rev_growth:.1%}")
                    elif rev_growth > 0.15:
                        raw_score += 2
                        details.append(f"Moderate revenue growth: {rev_growth:.1%}")
                    elif rev_growth > 0.05:
                        raw_score += 1
                        details.append(f"Slight revenue growth: {rev_growth:.1%}")
                    else:
                        details.append(f"Minimal/negative revenue growth: {rev_growth:.1%}")
                else:
                    details.append("Older revenue is zero/negative; can't compute revenue growth.")
            else:
                details.append("Not enough revenue data points for growth calculation.")
            break
    
    if not revenue_found:
        if not fundamental_data:
            details.append("No fundamental data available for revenue growth analysis.")
        elif "profit_loss" not in fundamental_data:
            details.append("No profit and loss data available for revenue growth analysis.")
        else:
            details.append("No revenue data found in profit and loss statement.")
    
    # 2. EPS Growth
    eps_found = False
    for eps_key in ["EPS", "Earnings Per Share", "Diluted EPS", "EPS in Rs"]:
        if fundamental_data and "profit_loss" in fundamental_data and eps_key in fundamental_data["profit_loss"]:
            eps_values = fundamental_data["profit_loss"][eps_key]
            eps_values = [float(e) if e is not None else None for e in eps_values]
            valid_eps = [e for e in eps_values if e is not None]
            eps_found = True
            
            if len(valid_eps) >= 2:
                latest_eps = valid_eps[0]
                older_eps = valid_eps[-1]
                # Avoid division by zero
                if abs(older_eps) > 1e-9:
                    eps_growth = (latest_eps - older_eps) / abs(older_eps)
                    if eps_growth > 0.30:
                        raw_score += 3
                        details.append(f"Strong EPS growth: {eps_growth:.1%}")
                    elif eps_growth > 0.15:
                        raw_score += 2
                        details.append(f"Moderate EPS growth: {eps_growth:.1%}")
                    elif eps_growth > 0.05:
                        raw_score += 1
                        details.append(f"Slight EPS growth: {eps_growth:.1%}")
                    else:
                        details.append(f"Minimal/negative EPS growth: {eps_growth:.1%}")
                else:
                    details.append("Older EPS is near zero; skipping EPS growth calculation.")
            else:
                details.append("Not enough EPS data points for growth calculation.")
            break
    
    if not eps_found:
        if not fundamental_data:
            details.append("No fundamental data available for EPS growth analysis.")
        elif "profit_loss" not in fundamental_data:
            details.append("No profit and loss data available for EPS growth analysis.")
        else:
            details.append("No EPS data found in profit and loss statement.")
    
    # 3. Price Momentum
    if prices_data is None:
        print("No price data provided for momentum analysis")
        details.append("No price data available for momentum analysis.")
    elif isinstance(prices_data, pd.DataFrame) and not prices_data.empty and 'close' in prices_data.columns and len(prices_data) >= 2:
        # Get first and last close price
        start_price = prices_data['close'].iloc[0]
        end_price = prices_data['close'].iloc[-1]
        
        if start_price > 0:
            pct_change = (end_price - start_price) / start_price
            if pct_change > 0.50:
                raw_score += 3
                details.append(f"Very strong price momentum: {pct_change:.1%}")
            elif pct_change > 0.20:
                raw_score += 2
                details.append(f"Moderate price momentum: {pct_change:.1%}")
            elif pct_change > 0:
                raw_score += 1
                details.append(f"Slight positive momentum: {pct_change:.1%}")
            else:
                details.append(f"Negative price momentum: {pct_change:.1%}")
        else:
            details.append("Invalid start price (<= 0); can't compute momentum.")
    elif hasattr(prices_data, 'closes') and len(prices_data.closes) >= 2:
        # For OHLCVData or similar objects
        start_price = prices_data.closes[0]
        end_price = prices_data.closes[-1]
        if start_price > 0:
            pct_change = (end_price - start_price) / start_price
            if pct_change > 0.50:
                raw_score += 3
                details.append(f"Very strong price momentum: {pct_change:.1%}")
            elif pct_change > 0.20:
                raw_score += 2
                details.append(f"Moderate price momentum: {pct_change:.1%}")
            elif pct_change > 0:
                raw_score += 1
                details.append(f"Slight positive momentum: {pct_change:.1%}")
            else:
                details.append(f"Negative price momentum: {pct_change:.1%}")
        else:
            details.append("Invalid start price (<= 0); can't compute momentum.")
    else:
        details.append("Insufficient price data for momentum calculation.")
    
    # Scale to 0–10
    final_score = min(10, (raw_score / 9) * 10)
    
    # Determine signal based on score
    if final_score >= 7.5:
        signal = "bullish"
    elif final_score <= 3.5:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_risk_reward(ticker: str, prices_data = None) -> Dict[str, Any]:
    """
    Assesses risk via:
      - Debt-to-Equity
      - Price Volatility
    Aims for strong upside with contained downside.
    
    Returns a dictionary with score and detailed analysis.
    """
    # Fetch fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    
    # Get market cap for valuation context
    market_cap = get_market_cap(ticker)
    
    details = []
    raw_score = 0  # We'll accumulate up to 6 raw points, then scale to 0-10
    
    # 1. Debt-to-Equity
    debt_found = False
    equity_found = False
    debt_equity_analyzed = False
    
    if fundamental_data and "balance_sheet" in fundamental_data:
        # First try to find debt in balance sheet
        for debt_key in ["Total Debt", "Long Term Debt", "Total Liabilities", "Long-term Borrowings", "Total Non-Current Liabilities"]:
            if debt_key in fundamental_data["balance_sheet"]:
                debt_values = fundamental_data["balance_sheet"][debt_key]
                debt_values = [float(d) if d is not None else None for d in debt_values]
                valid_debt = [d for d in debt_values if d is not None]
                
                if valid_debt:
                    recent_debt = valid_debt[0]
                    debt_found = True
                    break
        
        # Now try to find equity
        for equity_key in ["Total Equity", "Shareholders' Equity", "Total Shareholders' Equity", "Share Capital and Reserves", "Total Equity\xa0"]:
            if equity_key in fundamental_data["balance_sheet"]:
                equity_values = fundamental_data["balance_sheet"][equity_key]
                equity_values = [float(e) if e is not None else None for e in equity_values]
                valid_equity = [e for e in equity_values if e is not None]
                
                if valid_equity:
                    recent_equity = valid_equity[0]
                    equity_found = True
                    break
        
        # Calculate debt-to-equity ratio if both found
        if debt_found and equity_found and recent_equity > 0:
            de_ratio = recent_debt / recent_equity
            debt_equity_analyzed = True
            
            if de_ratio < 0.3:
                raw_score += 3
                details.append(f"Low debt-to-equity: {de_ratio:.2f}")
            elif de_ratio < 0.7:
                raw_score += 2
                details.append(f"Moderate debt-to-equity: {de_ratio:.2f}")
            elif de_ratio < 1.5:
                raw_score += 1
                details.append(f"Somewhat high debt-to-equity: {de_ratio:.2f}")
            else:
                details.append(f"High debt-to-equity: {de_ratio:.2f}")
    
    if not debt_equity_analyzed:
        if not fundamental_data or "balance_sheet" not in fundamental_data:
            details.append("No balance sheet data available for debt/equity analysis.")
        else:
            details.append("No consistent debt/equity data available.")
    
    # 2. Price Volatility
    if prices_data is None:
        print("No price data provided for volatility analysis")
        details.append("No price data available for volatility analysis.")
    elif isinstance(prices_data, pd.DataFrame) and not prices_data.empty and 'close' in prices_data.columns and len(prices_data) > 10:
        # Calculate daily returns
        prices_data['return'] = prices_data['close'].pct_change()
        daily_returns = prices_data['return'].dropna().tolist()
        
        if daily_returns:
            stdev = statistics.pstdev(daily_returns)  # population stdev
            if stdev < 0.01:
                raw_score += 3
                details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
            elif stdev < 0.02:
                raw_score += 2
                details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
            elif stdev < 0.04:
                raw_score += 1
                details.append(f"High volatility: daily returns stdev {stdev:.2%}")
            else:
                details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
        else:
            details.append("Insufficient daily returns data for volatility calc.")
    elif hasattr(prices_data, 'closes') and len(prices_data.closes) > 10:
        # For OHLCVData or similar objects
        daily_returns = []
        for i in range(1, len(prices_data.closes)):
            prev_close = prices_data.closes[i - 1]
            if prev_close > 0:
                daily_returns.append((prices_data.closes[i] - prev_close) / prev_close)
        
        if daily_returns:
            stdev = statistics.pstdev(daily_returns)  # population stdev
            if stdev < 0.01:
                raw_score += 3
                details.append(f"Low volatility: daily returns stdev {stdev:.2%}")
            elif stdev < 0.02:
                raw_score += 2
                details.append(f"Moderate volatility: daily returns stdev {stdev:.2%}")
            elif stdev < 0.04:
                raw_score += 1
                details.append(f"High volatility: daily returns stdev {stdev:.2%}")
            else:
                details.append(f"Very high volatility: daily returns stdev {stdev:.2%}")
        else:
            details.append("Insufficient daily returns data for volatility calc.")
    else:
        details.append("Not enough price data for volatility analysis.")
    
    # raw_score out of 6 => scale to 0–10
    final_score = min(10, (raw_score / 6) * 10)
    
    # Determine signal based on score
    if final_score >= 7.5:
        signal = "bullish"
    elif final_score <= 3.5:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": final_score,
        "details": "; ".join(details)
    }


def analyze_druckenmiller_valuation(ticker: str) -> Dict[str, Any]:
    """
    Druckenmiller is willing to pay up for growth, but still checks:
      - P/E
      - P/FCF
      - EV/EBIT
      - EV/EBITDA
    Each can yield up to 2 points => max 8 raw points => scale to 0–10.
    
    Returns a dictionary with score and detailed analysis.
    """
    # Fetch fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    
    # Get market cap for valuation metrics
    market_cap = get_market_cap(ticker)
    
    if not fundamental_data or market_cap is None:
        details = []
        if not fundamental_data:
            details.append("No fundamental data available for valuation analysis.")
        if market_cap is None:
            details.append("No market cap data available.")
        
        return {
            "signal": "neutral",
            "score": 0,
            "details": "; ".join(details)
        }
    
    details = []
    raw_score = 0
    
    # 1) P/E Ratio
    pe_data = get_pe_ratio(ticker)
    if pe_data and isinstance(pe_data, dict) and "current_pe" in pe_data:
        pe = pe_data["current_pe"]
        if pe < 15:
            raw_score += 2
            details.append(f"Attractive P/E: {pe:.2f}")
        elif pe < 25:
            raw_score += 1
            details.append(f"Fair P/E: {pe:.2f}")
        else:
            details.append(f"High or Very high P/E: {pe:.2f}")
    else:
        # Try to calculate P/E manually from net income
        net_income_found = False
        if "profit_loss" in fundamental_data:
            for ni_key in ["Net Income", "Net Profit", "Profit After Tax"]:
                if ni_key in fundamental_data["profit_loss"]:
                    net_income_values = fundamental_data["profit_loss"][ni_key]
                    net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
                    valid_ni = [ni for ni in net_income_values if ni is not None]
                    
                    if valid_ni:
                        recent_net_income = valid_ni[0]
                        net_income_found = True
                        
                        if recent_net_income > 0:
                            pe = market_cap / recent_net_income
                            if pe < 15:
                                raw_score += 2
                                details.append(f"Attractive P/E: {pe:.2f}")
                            elif pe < 25:
                                raw_score += 1
                                details.append(f"Fair P/E: {pe:.2f}")
                            else:
                                details.append(f"High or Very high P/E: {pe:.2f}")
                        else:
                            details.append("No positive net income for P/E calculation")
                        break
        
        if not net_income_found:
            details.append("No positive net income for P/E calculation")
    
    # 2) P/FCF
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "current_fcf" in fcf_data and fcf_data["current_fcf"] is not None:
        recent_fcf = fcf_data["current_fcf"]
        if recent_fcf > 0:
            pfcf = market_cap / recent_fcf
            if pfcf < 15:
                raw_score += 2
                details.append(f"Attractive P/FCF: {pfcf:.2f}")
            elif pfcf < 25:
                raw_score += 1
                details.append(f"Fair P/FCF: {pfcf:.2f}")
            else:
                details.append(f"High/Very high P/FCF: {pfcf:.2f}")
        else:
            details.append("No positive free cash flow for P/FCF calculation")
    else:
        # Try to get FCF from cash flow statement
        fcf_found = False
        if "cash_flows" in fundamental_data:
            for fcf_key in ["Free Cash Flow", "FCF"]:
                if fcf_key in fundamental_data["cash_flows"]:
                    fcf_values = fundamental_data["cash_flows"][fcf_key]
                    fcf_values = [float(f) if f is not None else None for f in fcf_values]
                    valid_fcf = [f for f in fcf_values if f is not None]
                    
                    if valid_fcf:
                        recent_fcf = valid_fcf[0]
                        fcf_found = True
                        
                        if recent_fcf > 0:
                            pfcf = market_cap / recent_fcf
                            if pfcf < 15:
                                raw_score += 2
                                details.append(f"Attractive P/FCF: {pfcf:.2f}")
                            elif pfcf < 25:
                                raw_score += 1
                                details.append(f"Fair P/FCF: {pfcf:.2f}")
                            else:
                                details.append(f"High/Very high P/FCF: {pfcf:.2f}")
                        else:
                            details.append("No positive free cash flow for P/FCF calculation")
                        break
        
        if not fcf_found:
            details.append("No positive free cash flow for P/FCF calculation")
    
    # Get enterprise value components
    recent_debt = 0
    recent_cash = 0
    
    # Find debt
    if "balance_sheet" in fundamental_data:
        for debt_key in ["Total Debt", "Long Term Debt"]:
            if debt_key in fundamental_data["balance_sheet"]:
                debt_values = fundamental_data["balance_sheet"][debt_key]
                debt_values = [float(d) if d is not None else None for d in debt_values]
                valid_debt = [d for d in debt_values if d is not None]
                
                if valid_debt:
                    recent_debt = valid_debt[0]
                    break
    
    # Find cash
    if "balance_sheet" in fundamental_data:
        for cash_key in ["Cash and Cash Equivalents", "Cash & Cash Equivalents", "Cash"]:
            if cash_key in fundamental_data["balance_sheet"]:
                cash_values = fundamental_data["balance_sheet"][cash_key]
                cash_values = [float(c) if c is not None else None for c in cash_values]
                valid_cash = [c for c in cash_values if c is not None]
                
                if valid_cash:
                    recent_cash = valid_cash[0]
                    break
    
    enterprise_value = market_cap + recent_debt - recent_cash
    
    # 3) EV/EBIT
    ebit_found = False
    if "profit_loss" in fundamental_data:
        for ebit_key in ["EBIT", "Operating Income", "Operating Profit"]:
            if ebit_key in fundamental_data["profit_loss"]:
                ebit_values = fundamental_data["profit_loss"][ebit_key]
                ebit_values = [float(e) if e is not None else None for e in ebit_values]
                valid_ebit = [e for e in ebit_values if e is not None]
                
                if valid_ebit:
                    recent_ebit = valid_ebit[0]
                    ebit_found = True
                    
                    if enterprise_value > 0 and recent_ebit > 0:
                        ev_ebit = enterprise_value / recent_ebit
                        if ev_ebit < 15:
                            raw_score += 2
                            details.append(f"Attractive EV/EBIT: {ev_ebit:.2f}")
                        elif ev_ebit < 25:
                            raw_score += 1
                            details.append(f"Fair EV/EBIT: {ev_ebit:.2f}")
                        else:
                            details.append(f"High EV/EBIT: {ev_ebit:.2f}")
                    else:
                        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")
                    break
    
    if not ebit_found:
        details.append("No valid EV/EBIT because EV <= 0 or EBIT <= 0")
    
    # 4) EV/EBITDA
    ebitda_found = False
    if "profit_loss" in fundamental_data:
        for ebitda_key in ["EBITDA", "EBITDA\xa0", "EBIDTA"]:
            if ebitda_key in fundamental_data["profit_loss"]:
                ebitda_values = fundamental_data["profit_loss"][ebitda_key]
                ebitda_values = [float(e) if e is not None else None for e in ebitda_values]
                valid_ebitda = [e for e in ebitda_values if e is not None]
                
                if valid_ebitda:
                    recent_ebitda = valid_ebitda[0]
                    ebitda_found = True
                    
                    if enterprise_value > 0 and recent_ebitda > 0:
                        ev_ebitda = enterprise_value / recent_ebitda
                        if ev_ebitda < 10:
                            raw_score += 2
                            details.append(f"Attractive EV/EBITDA: {ev_ebitda:.2f}")
                        elif ev_ebitda < 18:
                            raw_score += 1
                            details.append(f"Fair EV/EBITDA: {ev_ebitda:.2f}")
                        else:
                            details.append(f"High EV/EBITDA: {ev_ebitda:.2f}")
                    else:
                        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")
                    break
    
    if not ebitda_found:
        details.append("No valid EV/EBITDA because EV <= 0 or EBITDA <= 0")
    
    # We have up to 2 points for each of the 4 metrics => 8 raw points max
    # Scale raw_score to 0–10
    final_score = min(10, (raw_score / 8) * 10)
    
    # Determine signal based on score
    if final_score >= 7.5:
        signal = "bullish"
    elif final_score <= 3.5:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": final_score,
        "details": "; ".join(details)
    }


class DruckenmillerAgnoAgent():
    """Agno-based agent implementing Stanley Druckenmiller's investing principles."""
    
    def __init__(self, model_name: str = "sufe-aiflm-lab_fin-r1", model_provider: str = "lmstudio"):
        self.model_name = model_name
        self.model_provider = model_provider
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the Agno agent with appropriate model."""
        self.agent = Agent(
            model=LMStudio(id=self.model_name, reasoning_effort="high"),
            markdown=True,
            reasoning=True,
        )
    
    def _add_error_to_results(self, results, ticker, error_message):
        """Helper method to add error information to results dictionary for a ticker."""
        results[ticker] = {"error": error_message}
    
    def _prepare_agent_prompt(self, ticker: str, analysis_summary: Dict[str, Any]) -> str:
        """
        Prepare the prompt for the LLM based on Stanley Druckenmiller's investing principles.
        """
        prompt_template = """You are a Stanley Druckenmiller AI agent, making investment decisions using his principles:

                1. Focus on growth and momentum as primary signals
                2. Look for asymmetric risk-reward opportunities with high upside potential 
                3. Pay attention to valuation but be willing to pay for high-growth, high-potential companies
                4. Concentrate investments in high-conviction ideas
                5. Be willing to make aggressive moves when conviction is high

                Please analyze {ticker} using the following data:

                Growth & Momentum Analysis:
                {growth_momentum}

                Risk/Reward Analysis:
                {risk_reward}

                Valuation Analysis:
                {valuation}

                Based on this analysis and Stanley Druckenmiller's investment philosophy, would you recommend investing in {ticker}?
                Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
                Structure your reasoning to follow Druckenmiller's concentrated investing approach.
                """
        return prompt_template.format(
            ticker=ticker,
            growth_momentum=json.dumps(analysis_summary["growth_momentum"], indent=2),
            risk_reward=json.dumps(analysis_summary["risk_reward"], indent=2),
            valuation=json.dumps(analysis_summary["valuation"], indent=2)
        )
    
    def _parse_response(self, response: str, ticker: str) -> DruckenmillerSignal:
        """Parse the LLM response into a structured signal."""
        try:
            # Check if response is a string or an object
            response_text = response
            if not isinstance(response, str):
                # If response is an object from Agno agent, convert it to string
                response_text = str(response)
            
            # Attempt to extract signal information from the response
            if "bullish" in response_text.lower():
                signal = "bullish"
            elif "bearish" in response_text.lower():
                signal = "bearish"
            else:
                signal = "neutral"
            
            # Extract confidence (simple heuristic)
            confidence_matches = re.findall(r"confidence[:\s]+(\d+\.?\d*)", response_text.lower())
            confidence = float(confidence_matches[0]) if confidence_matches else 0.5
            
            # Ensure confidence is between 0 and 1
            if confidence > 1.0:
                confidence = confidence / 100.0 if confidence > 10 else 1.0
            
            # Truncate reasoning to reasonable length
            reasoning = response_text[:1000] if len(response_text) > 1000 else response_text
            
            return DruckenmillerSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
            return DruckenmillerSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Stanley Druckenmiller's investing principles:
          - Seeking asymmetric risk-reward opportunities
          - Emphasizing growth, momentum
          - Willing to be aggressive if conditions are favorable
          - Focus on preserving capital by avoiding high-risk, low-reward bets
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")
            
        agent_name = "druckenmiller_agno_agent"
        
        data = state.get("data", {})
        metadata = state.get("metadata", {})
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        tickers = data.get("tickers", [])
        
        if not tickers:
            return {f"{agent_name}_error": "Missing 'tickers' in input state."}
        
        # Update model if needed
        model_name = metadata.get("model_name", self.model_name)
        model_provider = metadata.get("model_provider", self.model_provider)
        if model_name != self.model_name or model_provider != self.model_provider:
            self.model_name = model_name
            self.model_provider = model_provider
            self._initialize_agent()
        
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker} with Stanley Druckenmiller's principles...")
                
                # Check if fundamental data is available
                fundamental_data = fetch_fundamental_data(ticker)
                if not fundamental_data:
                    print(f"No fundamental data found for {ticker}")
                    results[ticker] = {"error": f"No fundamental data found for {ticker}"}
                    continue

                # Get filtered price data using the tool function - will be required for analysis
                df = load_filtered_price_data(ticker, start_date, end_date)
                
                if df is None:
                    print(f"No price data found for {ticker}")
                    results[ticker] = {"error": f"Could not fetch price data for {ticker}"}
                    continue
                                
                # Debug info about available data
                data_availability = {
                    "has_price_data": df is not None,
                    "has_fundamental_data": fundamental_data is not None,
                    "has_profit_loss": "profit_loss" in fundamental_data if fundamental_data else False,
                    "has_balance_sheet": "balance_sheet" in fundamental_data if fundamental_data else False,
                    "has_cash_flows": "cash_flows" in fundamental_data if fundamental_data else False,
                    "has_market_data": "market_data" in fundamental_data if fundamental_data else False,
                    "price_data_points": len(df) if df is not None else 0
                }
                print(f"Data availability for {ticker}: {data_availability}")
                
                # Check if we have enough data to proceed with analysis
                if not data_availability["has_price_data"] or not data_availability["has_fundamental_data"]:
                    error_message = "Insufficient data for complete analysis: "
                    error_details = []
                    if not data_availability["has_price_data"]:
                        error_details.append("No price data available")
                    if not data_availability["has_fundamental_data"]:
                        error_details.append("No fundamental data available")
                    
                    error_message += ", ".join(error_details)
                    results[ticker] = {
                        "error": error_message,
                        "data_availability": data_availability
                    }
                    continue
                
                # Convert DataFrame to price data structure using the global function
                price_data = convert_df_to_price_data(df, ticker)
                
                # Perform analysis using Druckenmiller's key metrics
                growth_momentum_analysis = analyze_growth_and_momentum(ticker, df)
                risk_reward_analysis = analyze_risk_reward(ticker, df)
                valuation_analysis = analyze_druckenmiller_valuation(ticker)
                
                # Combine partial scores with adjusted weights:
                # 45% Growth/Momentum, 30% Risk/Reward, 25% Valuation
                total_score = (
                    growth_momentum_analysis["score"] * 0.45
                    + risk_reward_analysis["score"] * 0.30
                    + valuation_analysis["score"] * 0.25
                )
                
                # Determine signal based on total score
                if total_score >= 7.5:
                    calculated_signal = "bullish"
                elif total_score <= 4.5:
                    calculated_signal = "bearish"
                else:
                    calculated_signal = "neutral"
                
                # Calculate confidence based on how far from neutral the score is
                if calculated_signal == "neutral":
                    confidence = 50
                else:
                    # Scale confidence based on distance from the neutral threshold (5.0)
                    # Maximum confidence (100%) would be at score 10 (bullish) or 0 (bearish)
                    distance_from_neutral = abs(total_score - 5.0)
                    confidence = min(100, distance_from_neutral * 20)  # 20% per point of difference
                
                # Format the analysis summary for LLM
                analysis_summary = {
                    "overall_score": round(total_score, 1),
                    "growth_momentum": {
                        "score": round(growth_momentum_analysis["score"], 1),
                        "signal": growth_momentum_analysis["signal"],
                        "details": growth_momentum_analysis["details"],
                        "weight": "45%"
                    },
                    "risk_reward": {
                        "score": round(risk_reward_analysis["score"], 1),
                        "signal": risk_reward_analysis["signal"],
                        "details": risk_reward_analysis["details"],
                        "weight": "30%"
                    },
                    "valuation": {
                        "score": round(valuation_analysis["score"], 1),
                        "signal": valuation_analysis["signal"],
                        "details": valuation_analysis["details"],
                        "weight": "25%"
                    }
                }
                print(analysis_summary)
                # Prepare prompt and run LLM
                prompt = self._prepare_agent_prompt(ticker, analysis_summary)
                llm_response = self.agent.run(prompt)
                
                # Parse LLM response into structured signal
                signal = self._parse_response(llm_response, ticker)
                
                # Store results
                results[ticker] = {
                    "signal": signal.signal,
                    "confidence": round(float(signal.confidence) * 100),  # Convert to percentage
                    "reasoning": signal.reasoning,
                    "analysis": analysis_summary,
                    "data_availability": data_availability
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {"error": f"Error analyzing {ticker}: {str(e)}"}
        
        return {agent_name: results}


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_state = {
        "data": {
            "tickers": ["COLPAL"],  # Reliance Industries ticker (confirmed to have data)
            "start_date": "2022-01-01",  # Optional start date
            "end_date": "2023-12-31"  # Optional end date
        }
    }
    try:
        agent = DruckenmillerAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure market data is properly set up.") 