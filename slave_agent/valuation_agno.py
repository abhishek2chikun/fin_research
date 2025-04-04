"""
Valuation Analysis Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
from pydantic import BaseModel
from typing_extensions import Literal

# Import Agno framework
from agno.agent import Agent

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools
from tools.fundamental import fetch_fundamental_data, get_market_cap, get_free_cash_flow
from tools.fundamental import get_earnings_growth, get_working_capital_change, get_net_income
from tools.fundamental import get_depreciation_amortization, get_capital_expenditure

# Pydantic model for the output signal
class ValuationSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: Dict[str, Any]


def calculate_owner_earnings_value(ticker: str) -> Dict[str, Any]:
    """
    Calculates the intrinsic value using Buffett's Owner Earnings method.

    Owner Earnings = Net Income
                    + Depreciation/Amortization
                    - Capital Expenditures
                    - Working Capital Changes

    Returns a dictionary with valuation details.
    """
    # Get required financial data
    net_income_data = get_net_income(ticker)
    depreciation_data = get_depreciation_amortization(ticker)
    capex_data = get_capital_expenditure(ticker)
    working_capital_data = get_working_capital_change(ticker)
    growth_data = get_earnings_growth(ticker)
    market_cap_data = get_market_cap(ticker)
    
    if not all([net_income_data, depreciation_data, capex_data, working_capital_data, growth_data, market_cap_data]):
        return {
            "signal": "neutral",
            "details": "Insufficient data for owner earnings valuation",
            "intrinsic_value": None,
            "market_cap": market_cap_data,
            "margin_of_safety": None
        }
    
    # Extract the values
    net_income = net_income_data.get("current_value", 0)
    depreciation = depreciation_data.get("current_value", 0)
    capex = capex_data.get("current_value", 0)
    working_capital_change = working_capital_data.get("current_change", 0)
    growth_rate = growth_data.get("growth_rate", 0.05)
    market_cap = market_cap_data
    
    # Set default values for calculation
    required_return = 0.15  # Required rate of return (Buffett typically uses 15%)
    margin_of_safety_factor = 0.25  # Margin of safety to apply to final value
    num_years = 5  # Number of years to project
    
    # Calculate initial owner earnings
    owner_earnings = net_income + depreciation - abs(capex) - working_capital_change
    
    if owner_earnings <= 0:
        return {
            "signal": "bearish",
            "details": f"Negative owner earnings: {owner_earnings:,.2f}",
            "intrinsic_value": 0,
            "market_cap": market_cap,
            "margin_of_safety": -1
        }
    
    # Project future owner earnings and discount them
    future_values = []
    for year in range(1, num_years + 1):
        future_value = owner_earnings * (1 + growth_rate) ** year
        discounted_value = future_value / (1 + required_return) ** year
        future_values.append(discounted_value)
    
    # Calculate terminal value (using perpetuity growth formula)
    terminal_growth = min(growth_rate, 0.03)  # Cap terminal growth at 3%
    terminal_value = (future_values[-1] * (1 + terminal_growth)) / (required_return - terminal_growth)
    terminal_value_discounted = terminal_value / (1 + required_return) ** num_years
    
    # Sum all values and apply margin of safety
    intrinsic_value = sum(future_values) + terminal_value_discounted
    value_with_safety_margin = intrinsic_value * (1 - margin_of_safety_factor)
    
    # Calculate margin of safety relative to market price
    margin_of_safety = (value_with_safety_margin - market_cap) / market_cap if market_cap > 0 else 0
    
    # Determine signal based on margin of safety
    signal = "neutral"
    if margin_of_safety > 0.15:  # More than 15% undervalued
        signal = "bullish"
    elif margin_of_safety < -0.15:  # More than 15% overvalued
        signal = "bearish"
    
    return {
        "signal": signal,
        "details": f"Owner Earnings Value: ${value_with_safety_margin:,.2f}, Market Cap: ${market_cap:,.2f}, Margin of Safety: {margin_of_safety:.1%}",
        "intrinsic_value": value_with_safety_margin,
        "market_cap": market_cap,
        "margin_of_safety": margin_of_safety
    }


def calculate_dcf_value(ticker: str) -> Dict[str, Any]:
    """
    Computes the discounted cash flow (DCF) for a given company based on free cash flow.
    
    Returns a dictionary with valuation details.
    """
    # Get required financial data
    fcf_data = get_free_cash_flow(ticker)
    growth_data = get_earnings_growth(ticker)
    market_cap_data = get_market_cap(ticker)
    
    # Check if we need to calculate FCF manually from components
    if not fcf_data or "current_fcf" not in fcf_data:
        # Try to calculate FCF from components
        net_income_data = get_net_income(ticker)
        depreciation_data = get_depreciation_amortization(ticker)
        capex_data = get_capital_expenditure(ticker)
        working_capital_data = get_working_capital_change(ticker)
        
        if (net_income_data and "current_value" in net_income_data and
            depreciation_data and "current_value" in depreciation_data and
            capex_data and "current_value" in capex_data):
            
            net_income = net_income_data.get("current_value", 0)
            depreciation = depreciation_data.get("current_value", 0)
            capex = capex_data.get("current_value", 0)
            wc_change = working_capital_data.get("current_change", 0) if working_capital_data else 0
            
            # Calculate FCF manually
            # FCF = Net Income + Depreciation - CapEx - Working Capital Change
            free_cash_flow = net_income + depreciation - capex - wc_change
            
            # Create a synthetic FCF data object
            fcf_data = {
                "current_fcf": free_cash_flow,
                "is_estimated": True,
                "source": "calculated_from_components"
            }
            print(f"Calculated FCF manually: {free_cash_flow}")
    
    if not all([fcf_data, growth_data, market_cap_data]):
        return {
            "signal": "neutral",
            "details": "Insufficient data for DCF valuation",
            "intrinsic_value": None,
            "market_cap": market_cap_data,
            "margin_of_safety": None
        }
    
    # Extract the values
    free_cash_flow = fcf_data.get("current_fcf", 0)
    growth_rate = growth_data.get("growth_rate", 0.05)
    market_cap = market_cap_data
    
    # Set default parameters for DCF calculation
    discount_rate = 0.10
    terminal_growth_rate = 0.03
    num_years = 5
    
    # Check if we have valid FCF
    if free_cash_flow <= 0:
        return {
            "signal": "bearish",
            "details": f"Negative free cash flow: {free_cash_flow:,.2f}",
            "intrinsic_value": 0,
            "market_cap": market_cap,
            "margin_of_safety": -1
        }
    
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]
    
    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)
    
    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years
    
    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value
    
    # Calculate margin of safety relative to market price
    margin_of_safety = (dcf_value - market_cap) / market_cap if market_cap > 0 else 0
    
    # Determine signal based on margin of safety
    signal = "neutral"
    if margin_of_safety > 0.15:  # More than 15% undervalued
        signal = "bullish"
    elif margin_of_safety < -0.15:  # More than 15% overvalued
        signal = "bearish"
    
    return {
        "signal": signal,
        "details": f"DCF Value: ${dcf_value:,.2f}, Market Cap: ${market_cap:,.2f}, Margin of Safety: {margin_of_safety:.1%}",
        "intrinsic_value": dcf_value,
        "market_cap": market_cap,
        "margin_of_safety": margin_of_safety
    }


class ValuationAgnoAgent():
    """Agno-based agent implementing valuation analysis."""
    
    def __init__(self):
        pass
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using valuation methodologies.
        """
        agent_name = "valuation_agno_agent"
        
        data = state.get("data", {})
        tickers = data.get("tickers", [])
        
        if not tickers:
            return {f"{agent_name}_error": "Missing 'tickers' in input state."}
        
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker} with valuation analysis...")
                
                # Check for fundamental data availability
                fundamental_data = fetch_fundamental_data(ticker)
                if not fundamental_data:
                    print(f"No fundamental data found for {ticker}")
                    results[ticker] = {
                        "signal": "neutral",
                        "confidence": 0,
                        "error": "No fundamental data available"
                    }
                    continue
                
                # Log available data sections for debugging
                data_availability = {
                    "profit_loss": "profit_loss" in fundamental_data,
                    "balance_sheet": "balance_sheet" in fundamental_data,
                    "cash_flows": "cash_flows" in fundamental_data,
                    "ratios": "ratios" in fundamental_data,
                    "market_data": "market_data" in fundamental_data,
                }
                print(f"Data availability for {ticker}: {data_availability}")
                
                # Fetch required financial data
                net_income_data = get_net_income(ticker)
                depreciation_data = get_depreciation_amortization(ticker)
                capex_data = get_capital_expenditure(ticker)
                working_capital_data = get_working_capital_change(ticker)
                growth_data = get_earnings_growth(ticker)
                market_cap_data = get_market_cap(ticker)
                fcf_data = get_free_cash_flow(ticker)
                
                # Log data retrieval results
                print(f"Net Income: {net_income_data.get('current_value')}")
                print(f"Depreciation: {depreciation_data.get('current_value')}")
                print(f"CapEx: {capex_data.get('current_value')}")
                print(f"Working Capital Change: {working_capital_data.get('current_change')}")
                print(f"Growth Rate: {growth_data.get('growth_rate')}")
                print(f"Market Cap: {market_cap_data}")
                print(f"Free Cash Flow: {fcf_data.get('current_fcf')}")
                
                # Perform valuation analysis using the modular functions
                owner_earnings_analysis = calculate_owner_earnings_value(ticker)
                dcf_analysis = calculate_dcf_value(ticker)
                
                # Check if both analyses failed
                if (owner_earnings_analysis.get("signal") == "neutral" and 
                    owner_earnings_analysis.get("details", "").startswith("Insufficient data") and
                    dcf_analysis.get("signal") == "neutral" and 
                    dcf_analysis.get("details", "").startswith("Insufficient data")):
                    
                    print(f"Insufficient data for both valuation methods for {ticker}")
                    results[ticker] = {
                        "signal": "neutral",
                        "confidence": 0,
                        "reasoning": {
                            "dcf_analysis": {
                                "signal": "neutral",
                                "details": "Insufficient data for DCF valuation",
                            },
                            "owner_earnings_analysis": {
                                "signal": "neutral",
                                "details": "Insufficient data for owner earnings valuation",
                            }
                        },
                        "error": "Insufficient financial data for valuation"
                    }
                    continue
                
                # Calculate combined valuation gap (average of both methods)
                oe_margin = owner_earnings_analysis.get("margin_of_safety")
                dcf_margin = dcf_analysis.get("margin_of_safety")
                
                # Check if we have valid margins from both methods
                if oe_margin is not None and dcf_margin is not None:
                    valuation_gap = (oe_margin + dcf_margin) / 2
                elif oe_margin is not None:
                    valuation_gap = oe_margin
                elif dcf_margin is not None:
                    valuation_gap = dcf_margin
                else:
                    valuation_gap = 0
                
                # Determine signal based on combined valuation gap
                if valuation_gap > 0.15:  # More than 15% undervalued
                    signal = "bullish"
                elif valuation_gap < -0.15:  # More than 15% overvalued
                    signal = "bearish"
                else:
                    signal = "neutral"
                
                # Calculate confidence based on magnitude of valuation gap
                confidence = min(abs(valuation_gap) * 100, 100)
                
                # Store results
                results[ticker] = {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": {
                        "dcf_analysis": {
                            "signal": dcf_analysis.get("signal", "neutral"),
                            "details": dcf_analysis.get("details", "DCF analysis not available"),
                            "intrinsic_value": dcf_analysis.get("intrinsic_value"),
                            "market_cap": dcf_analysis.get("market_cap"),
                            "margin_of_safety": dcf_analysis.get("margin_of_safety")
                        },
                        "owner_earnings_analysis": {
                            "signal": owner_earnings_analysis.get("signal", "neutral"),
                            "details": owner_earnings_analysis.get("details", "Owner earnings analysis not available"),
                            "intrinsic_value": owner_earnings_analysis.get("intrinsic_value"),
                            "market_cap": owner_earnings_analysis.get("market_cap"),
                            "margin_of_safety": owner_earnings_analysis.get("margin_of_safety")
                        }
                    },
                    "valuation_gap": valuation_gap,
                    "data_availability": data_availability
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {
                    "signal": "neutral", 
                    "confidence": 0,
                    "error": f"Error analyzing {ticker}: {str(e)}"
                }
        
        return {agent_name: results}


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_state = {
        "data": {
            "tickers": ["COLPAL"],  # Example ticker
            "end_date": "2023-12-31"  # Optional end date
        }
    }
    try:
        agent = ValuationAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 