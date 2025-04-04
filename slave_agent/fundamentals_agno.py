"""
Fundamental Analysis Agent using Agno Framework
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
from tools.fundamental import fetch_fundamental_data, get_financial_metrics
from tools.fundamental import get_pe_ratio, get_return_on_equity, get_operating_margin
from tools.fundamental import get_free_cash_flow, get_current_ratio, get_debt_to_equity
from tools.fundamental import get_revenue_growth, get_earnings_growth, get_book_value_growth
from tools.fundamental import get_price_to_book_ratio, get_price_to_sales_ratio

# Pydantic model for the output signal
class FundamentalsSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: Dict[str, Any]


def analyze_profitability(ticker: str) -> dict:
    """
    Analyze profitability metrics including ROE, net margin, and operating margin.
    """
    roe_data = get_return_on_equity(ticker)
    operating_margin_data = get_operating_margin(ticker)
    
    if not roe_data or not operating_margin_data:
        return {
            "signal": "neutral",
            "score": 0,
            "details": "Insufficient data to analyze profitability"
        }
    
    # Initialize counters and details
    profitability_score = 0
    details = []
    
    # ROE Analysis
    if "current_roe" in roe_data and roe_data["current_roe"] is not None:
        roe = roe_data["current_roe"]
        if roe > 0.15:  # Strong ROE above 15%
            profitability_score += 1
            details.append(f"ROE: {roe:.2%}")
        else:
            details.append(f"ROE: {roe:.2%}")
    else:
        details.append("ROE: N/A")
    
    # Operating Margin Analysis
    if "current_operating_margin" in operating_margin_data and operating_margin_data["current_operating_margin"] is not None:
        operating_margin = operating_margin_data["current_operating_margin"]
        if operating_margin > 0.15:  # Strong operating efficiency
            profitability_score += 1
            details.append(f"Operating Margin: {operating_margin:.2%}")
        else:
            details.append(f"Operating Margin: {operating_margin:.2%}")
    else:
        details.append("Operating Margin: N/A")
    
    # Net Margin Analysis (try from fundamental data if specific utility doesn't exist)
    net_margin = None
    fundamental_data = fetch_fundamental_data(ticker)
    if fundamental_data and "ratios" in fundamental_data and "Net Profit Margin %" in fundamental_data["ratios"]:
        net_margin_values = [m/100 if m and m > 1 else m for m in fundamental_data["ratios"]["Net Profit Margin %"] if m is not None]
        if net_margin_values:
            net_margin = net_margin_values[-1]  # Get most recent
    
    if net_margin is not None:
        if net_margin > 0.20:  # Healthy profit margins
            profitability_score += 1
            details.append(f"Net Margin: {net_margin:.2%}")
        else:
            details.append(f"Net Margin: {net_margin:.2%}")
    else:
        details.append("Net Margin: N/A")
    
    # Determine signal based on profitability score
    if profitability_score >= 2:
        signal = "bullish"
    elif profitability_score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": profitability_score,
        "details": ", ".join(details)
    }


def analyze_growth(ticker: str) -> dict:
    """
    Analyze growth metrics including revenue growth, earnings growth, and book value growth.
    """
    revenue_growth_data = get_revenue_growth(ticker)
    earnings_growth_data = get_earnings_growth(ticker)
    book_value_growth_data = get_book_value_growth(ticker)
    
    # Initialize counters and details
    growth_score = 0
    details = []
    
    # Revenue Growth Analysis
    revenue_growth = None
    if revenue_growth_data and "growth_rate" in revenue_growth_data:
        revenue_growth = revenue_growth_data["growth_rate"]
        if revenue_growth > 0.10:  # 10% growth threshold
            growth_score += 1
            details.append(f"Revenue Growth: {revenue_growth:.2%}")
        else:
            details.append(f"Revenue Growth: {revenue_growth:.2%}")
    else:
        details.append("Revenue Growth: N/A")
    
    # Earnings Growth Analysis
    earnings_growth = None
    if earnings_growth_data and "growth_rate" in earnings_growth_data:
        earnings_growth = earnings_growth_data["growth_rate"]
        if earnings_growth > 0.10:  # 10% growth threshold
            growth_score += 1
            details.append(f"Earnings Growth: {earnings_growth:.2%}")
        else:
            details.append(f"Earnings Growth: {earnings_growth:.2%}")
    else:
        details.append("Earnings Growth: N/A")
    
    # Book Value Growth Analysis
    book_value_growth = None
    if book_value_growth_data and "growth_rate" in book_value_growth_data:
        book_value_growth = book_value_growth_data["growth_rate"]
        if book_value_growth > 0.10:  # 10% growth threshold
            growth_score += 1
            details.append(f"Book Value Growth: {book_value_growth:.2%}")
        else:
            details.append(f"Book Value Growth: {book_value_growth:.2%}")
    else:
        details.append("Book Value Growth: N/A")
    
    # Determine signal based on growth score
    if growth_score >= 2:
        signal = "bullish"
    elif growth_score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": growth_score,
        "details": ", ".join(details)
    }


def analyze_financial_health(ticker: str) -> dict:
    """
    Analyze financial health metrics including current ratio, debt-to-equity, and free cash flow.
    """
    current_ratio_data = get_current_ratio(ticker)
    debt_equity_data = get_debt_to_equity(ticker)
    fcf_data = get_free_cash_flow(ticker)
    
    # Initialize counters and details
    health_score = 0
    details = []
    
    # Current Ratio Analysis
    if current_ratio_data and "current_ratio" in current_ratio_data:
        current_ratio = current_ratio_data["current_ratio"]
        if current_ratio > 1.5:  # Strong liquidity
            health_score += 1
            details.append(f"Current Ratio: {current_ratio:.2f}")
        else:
            details.append(f"Current Ratio: {current_ratio:.2f}")
    else:
        details.append("Current Ratio: N/A")
    
    # Debt-to-Equity Analysis
    if debt_equity_data and "current_ratio" in debt_equity_data:
        debt_to_equity = debt_equity_data["current_ratio"]
        if debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
            details.append(f"D/E: {debt_to_equity:.2f}")
        else:
            details.append(f"D/E: {debt_to_equity:.2f}")
    else:
        details.append("D/E: N/A")
    
    # Free Cash Flow Analysis
    if fcf_data and "current_fcf" in fcf_data and "eps" in fcf_data:
        fcf = fcf_data["current_fcf"]
        eps = fcf_data.get("eps")
        
        # Check FCF/EPS ratio if both are available
        if fcf and eps and eps != 0:
            fcf_to_eps = fcf / eps
            if fcf_to_eps > 0.8:  # Strong FCF conversion relative to earnings
                health_score += 1
                details.append(f"FCF/EPS: {fcf_to_eps:.2f}")
            else:
                details.append(f"FCF/EPS: {fcf_to_eps:.2f}")
        # Otherwise just check if FCF is positive
        elif fcf > 0:
            health_score += 1
            details.append(f"Positive FCF: {fcf:,.2f}")
        else:
            details.append(f"FCF: {fcf:,.2f}")
    else:
        details.append("FCF: N/A")
    
    # Determine signal based on financial health score
    if health_score >= 2:
        signal = "bullish"
    elif health_score == 0:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "signal": signal,
        "score": health_score,
        "details": ", ".join(details)
    }


def analyze_valuation_ratios(ticker: str) -> dict:
    """
    Analyze valuation ratios including P/E, P/B, and P/S ratios.
    """
    pe_data = get_pe_ratio(ticker)
    pb_data = get_price_to_book_ratio(ticker)
    ps_data = get_price_to_sales_ratio(ticker)
    
    # Initialize counters and details
    valuation_score = 0
    details = []
    
    # For valuation metrics, we count overvalued metrics, and signal is inversely related
    # (more overvalued metrics = bearish signal)
    
    # P/E Ratio Analysis
    if pe_data and "current_pe" in pe_data:
        pe_ratio = pe_data["current_pe"]
        if pe_ratio > 25:  # Potentially overvalued
            valuation_score += 1
            details.append(f"P/E: {pe_ratio:.2f}")
        else:
            details.append(f"P/E: {pe_ratio:.2f}")
    else:
        details.append("P/E: N/A")
    
    # P/B Ratio Analysis
    if pb_data and "current_pb" in pb_data:
        pb_ratio = pb_data["current_pb"]
        if pb_ratio > 3:  # Potentially overvalued
            valuation_score += 1
            details.append(f"P/B: {pb_ratio:.2f}")
        else:
            details.append(f"P/B: {pb_ratio:.2f}")
    else:
        details.append("P/B: N/A")
    
    # P/S Ratio Analysis
    if ps_data and "current_ps" in ps_data:
        ps_ratio = ps_data["current_ps"]
        if ps_ratio > 5:  # Potentially overvalued
            valuation_score += 1
            details.append(f"P/S: {ps_ratio:.2f}")
        else:
            details.append(f"P/S: {ps_ratio:.2f}")
    else:
        details.append("P/S: N/A")
    
    # Determine signal based on valuation score (inverse relationship)
    if valuation_score >= 2:
        signal = "bearish"  # Overvalued on multiple metrics
    elif valuation_score == 0:
        signal = "bullish"  # Not overvalued on any metrics
    else:
        signal = "neutral"  # Mixed valuation metrics
    
    return {
        "signal": signal,
        "score": valuation_score,
        "details": ", ".join(details)
    }


class FundamentalsAgnoAgent():
    """Agno-based agent implementing fundamental analysis."""
    
    def __init__(self):
        pass
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using fundamental analysis.
        """
        agent_name = "fundamentals_agno_agent"
        
        data = state.get("data", {})
        tickers = data.get("tickers", [])
        
        if not tickers:
            return {f"{agent_name}_error": "Missing 'tickers' in input state."}
        
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker} with fundamental analysis...")
                
                # Perform fundamental analysis using the modular functions
                profitability = analyze_profitability(ticker)
                growth = analyze_growth(ticker)
                financial_health = analyze_financial_health(ticker)
                valuation_ratios = analyze_valuation_ratios(ticker)
                
                # Collect all signals
                signals = [
                    profitability["signal"],
                    growth["signal"],
                    financial_health["signal"],
                    valuation_ratios["signal"]
                ]
                
                # Count bullish and bearish signals
                bullish_signals = signals.count("bullish")
                bearish_signals = signals.count("bearish")
                
                # Determine overall signal
                if bullish_signals > bearish_signals:
                    overall_signal = "bullish"
                elif bearish_signals > bullish_signals:
                    overall_signal = "bearish"
                else:
                    overall_signal = "neutral"
                
                # Calculate confidence level
                total_signals = len(signals)
                confidence = max(bullish_signals, bearish_signals) / total_signals
                
                # Store results
                results[ticker] = {
                    "signal": overall_signal,
                    "confidence": confidence * 100,  # Convert to percentage
                    "reasoning": {
                        "profitability_signal": {
                            "signal": profitability["signal"],
                            "details": profitability["details"],
                        },
                        "growth_signal": {
                            "signal": growth["signal"],
                            "details": growth["details"],
                        },
                        "financial_health_signal": {
                            "signal": financial_health["signal"],
                            "details": financial_health["details"],
                        },
                        "price_ratios_signal": {
                            "signal": valuation_ratios["signal"],
                            "details": valuation_ratios["details"],
                        }
                    }
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
            "tickers": ["AAPL"],  # Example ticker
            "end_date": "2023-12-31"  # Optional end date
        }
    }
    try:
        agent = FundamentalsAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 