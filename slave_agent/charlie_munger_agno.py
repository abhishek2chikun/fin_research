"""
Charlie Munger Investing Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
import math
import re
from pydantic import BaseModel
from typing_extensions import Literal

# Import Agno framework
from agno.agent import Agent
from agno.models.lmstudio import LMStudio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools
from tools.fundamental import fetch_fundamental_data, get_market_cap, get_revenue_growth, get_roe
from tools.fundamental import get_pe_ratio, get_operating_margin, get_debt_to_equity, get_free_cash_flow

# Pydantic model for the output signal
class CharlieMungerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def analyze_moat_strength(ticker: str) -> dict:
    """
    Analyze the business's competitive advantage using Munger's approach:
    - Consistent high returns on capital (ROIC)
    - Pricing power (stable/improving gross margins)
    - Low capital requirements
    - Network effects and intangible assets (R&D investments, goodwill)
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"score": 0, "details": "Insufficient data to analyze moat strength", "rating": "Poor", "summary": "Moat Strength: Poor (0/10 points). Insufficient data."}
        
    # 1. Return on Invested Capital (ROIC) analysis - Munger's favorite metric
    # Try to get ROE as a proxy for ROIC if direct ROIC not available
    roe_data = get_roe(ticker)
    
    if roe_data:
        roe_values = roe_data.get("values", [])
        if roe_values:
            # Check if ROE consistently above 15% (Munger's threshold)
            high_roe_count = sum(1 for r in roe_values if r is not None and r > 0.15)
            if high_roe_count >= len(roe_values) * 0.8:  # 80% of periods show high ROE
                score += 3
                details.append(f"Excellent ROE: >15% in {high_roe_count}/{len(roe_values)} periods")
            elif high_roe_count >= len(roe_values) * 0.5:  # 50% of periods
                score += 2
                details.append(f"Good ROE: >15% in {high_roe_count}/{len(roe_values)} periods")
            elif high_roe_count > 0:
                score += 1
                details.append(f"Mixed ROE: >15% in only {high_roe_count}/{len(roe_values)} periods")
            else:
                details.append("Poor ROE: Never exceeds 15% threshold")
        else:
            details.append("No ROE values available")
    else:
        details.append("No ROE data available")
    
    # 2. Pricing power - check gross margin stability and trends
    profit_loss = fundamental_data.get("profit_loss", {})
    gross_margins = []
    
    # Look for gross margin in profit_loss data
    for margin_key in ["Gross Profit Margin %", "GPM %", "Gross Margin %"]:
        if margin_key in profit_loss:
            gross_margins = profit_loss[margin_key]
            gross_margins = [float(gm)/100 if gm is not None and float(gm) > 1 else float(gm) if gm is not None else None for gm in gross_margins]
            break
    
    if gross_margins and len(gross_margins) >= 3:
        # Remove None values
        gross_margins = [gm for gm in gross_margins if gm is not None]
        
        if len(gross_margins) >= 3:
            # Munger likes stable or improving gross margins
            margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
            if margin_trend >= len(gross_margins) * 0.7:  # Improving in 70% of periods
                score += 2
                details.append("Strong pricing power: Gross margins consistently improving")
            elif sum(gross_margins) / len(gross_margins) > 0.3:  # Average margin > 30%
                score += 1
                details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
            else:
                details.append("Limited pricing power: Low or declining gross margins")
        else:
            details.append("Insufficient gross margin data after filtering")
    else:
        details.append("Insufficient gross margin data")
    
    # 3. Capital intensity - Munger prefers low capex businesses
    cash_flows = fundamental_data.get("cash_flows", {})
    capex_values = []
    revenue_values = []
    
    # Get capital expenditure values
    for capex_key in ["Capital Expenditure", "CAPEX", "Purchase of Fixed Assets"]:
        if capex_key in cash_flows:
            capex_values = cash_flows[capex_key]
            capex_values = [float(c) if c is not None else None for c in capex_values]
            break
    
    # Get revenue values
    if "Revenue" in profit_loss:
        revenue_values = profit_loss["Revenue"]
        revenue_values = [float(r) if r is not None else None for r in revenue_values]
    
    # Calculate capex to revenue ratio
    if capex_values and revenue_values and len(capex_values) >= 3 and len(revenue_values) >= 3:
        # Use minimum length to align the arrays
        min_length = min(len(capex_values), len(revenue_values))
        capex_to_revenue = []
        
        for i in range(min_length):
            if capex_values[i] is not None and revenue_values[i] is not None and revenue_values[i] > 0:
                # Note: capital_expenditure is typically negative in financial statements
                capex_ratio = abs(capex_values[i]) / revenue_values[i]
                capex_to_revenue.append(capex_ratio)
        
        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:  # Less than 5% of revenue
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:  # Less than 10% of revenue
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            details.append("Unable to calculate capital intensity ratio")
    else:
        details.append("Insufficient data for capital intensity analysis")
    
    # 4. Intangible assets - Munger values R&D and intellectual property
    balance_sheet = fundamental_data.get("balance_sheet", {})
    r_and_d_values = []
    goodwill_values = []
    
    # Look for R&D expenses
    if "Research and Development" in profit_loss:
        r_and_d_values = profit_loss["Research and Development"]
        r_and_d_values = [float(rd) if rd is not None else None for rd in r_and_d_values]
    
    # Look for goodwill and intangible assets
    for goodwill_key in ["Goodwill", "Intangible Assets", "Goodwill and Intangible Assets"]:
        if goodwill_key in balance_sheet:
            goodwill_values = balance_sheet[goodwill_key]
            goodwill_values = [float(gw) if gw is not None else None for gw in goodwill_values]
            break
    
    if r_and_d_values and any(rd is not None and rd > 0 for rd in r_and_d_values):
        score += 1
        details.append("Invests in R&D, building intellectual property")
    
    if goodwill_values and any(gw is not None and gw > 0 for gw in goodwill_values):
        score += 1
        details.append("Significant goodwill/intangible assets, suggesting brand value or IP")
    
    # 5. Operating margins - Munger values high and consistent operating margins
    operating_margin_data = get_operating_margin(ticker)
    if operating_margin_data and "historical_margins" in operating_margin_data:
        margins = operating_margin_data["historical_margins"]
        if operating_margin_data.get("is_consistently_high"):
            score += 2
            details.append("High and consistent operating margins indicate strong moat")
        elif operating_margin_data.get("current_operating_margin") and operating_margin_data.get("current_operating_margin") > 0.15:
            score += 1
            details.append(f"Current operating margin of {operating_margin_data['current_operating_margin']*100:.1f}% shows some pricing power")
        else:
            details.append("Operating margins suggest limited competitive advantages")
    else:
        details.append("No operating margin data available")
    
    # Scale score to 0-10 range
    max_raw_score = 9  # Max possible raw score (3+2+2+1+1)
    final_score = min(10, score * 10 / max_raw_score)
    
    # Determine rating based on score
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Moat Strength: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Strong competitive advantages with durable moat."
    elif final_score >= 5:
        summary += ". Good business economics with some defensible advantages."
    elif final_score >= 2.5:
        summary += ". Limited competitive advantages."
    else:
        summary += ". Weak moat with few competitive advantages."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    }


def analyze_management_quality(ticker: str) -> dict:
    """
    Evaluate management quality using Munger's criteria:
    - Capital allocation wisdom
    - Insider ownership and transactions
    - Cash management efficiency
    - Long-term focus
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"score": 0, "details": "Insufficient data to analyze management quality", "rating": "Poor", "summary": "Management Quality: Poor (0/10 points). Insufficient data."}
    
    # 1. Capital allocation - Check FCF to net income ratio
    # Munger values companies that convert earnings to cash
    fcf_data = get_free_cash_flow(ticker)
    
    if fcf_data and "historical_fcf" in fcf_data:
        fcf_values = fcf_data["historical_fcf"]
        
        # Get net income values
        profit_loss = fundamental_data.get("profit_loss", {})
        net_income_values = []
        
        if "Net Profit" in profit_loss:
            net_income_values = profit_loss["Net Profit"]
            net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
        
        if fcf_values and net_income_values and len(fcf_values) > 0 and len(net_income_values) > 0:
            # Calculate FCF to Net Income ratio for each period
            # Use the minimum length to align the arrays
            min_length = min(len(fcf_values), len(net_income_values))
            fcf_to_ni_ratios = []
            
            for i in range(min_length):
                if (fcf_values[i] is not None and net_income_values[i] is not None 
                    and net_income_values[i] > 0):
                    fcf_to_ni_ratios.append(fcf_values[i] / net_income_values[i])
            
            if fcf_to_ni_ratios:
                avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
                if avg_ratio > 1.1:  # FCF > net income suggests good accounting
                    score += 3
                    details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.9:  # FCF roughly equals net income
                    score += 2
                    details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                elif avg_ratio > 0.7:  # FCF somewhat lower than net income
                    score += 1
                    details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
                else:
                    details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
            else:
                details.append("Could not calculate FCF to Net Income ratios")
        else:
            details.append("Missing FCF or Net Income data")
    else:
        details.append("Free cash flow data not available")
    
    # 2. Debt management - Munger is cautious about debt
    debt_equity_data = get_debt_to_equity(ticker)
    
    if debt_equity_data and "current_ratio" in debt_equity_data:
        recent_de_ratio = debt_equity_data["current_ratio"]
        
        if recent_de_ratio < 0.3:  # Very low debt
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:  # Moderate debt
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:  # Higher but still reasonable debt
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Debt to equity data not available")
    
    # 3. Cash management efficiency - Munger values appropriate cash levels
    balance_sheet = fundamental_data.get("balance_sheet", {})
    profit_loss = fundamental_data.get("profit_loss", {})
    cash_values = []
    revenue_values = []
    
    # Get cash and equivalents
    for cash_key in ["Cash and Equivalents", "Cash & Bank Balance", "Cash and Bank Balances"]:
        if cash_key in balance_sheet:
            cash_values = balance_sheet[cash_key]
            cash_values = [float(c) if c is not None else None for c in cash_values]
            break
    
    # Get revenue values
    if "Revenue" in profit_loss:
        revenue_values = profit_loss["Revenue"]
        revenue_values = [float(r) if r is not None else None for r in revenue_values]
    
    if cash_values and revenue_values and len(cash_values) > 0 and len(revenue_values) > 0:
        # Get latest values
        latest_cash = cash_values[-1]
        latest_revenue = revenue_values[-1]
        
        if latest_cash is not None and latest_revenue is not None and latest_revenue > 0:
            # Calculate cash to revenue ratio (Munger likes 10-20% for most businesses)
            cash_to_revenue = latest_cash / latest_revenue
            
            if 0.1 <= cash_to_revenue <= 0.25:
                # Goldilocks zone - not too much, not too little
                score += 2
                details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
                # Reasonable but not ideal
                score += 1
                details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            elif cash_to_revenue > 0.4:
                # Too much cash - potentially inefficient capital allocation
                details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
            else:
                # Too little cash - potentially risky
                details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            details.append("Unable to calculate cash to revenue ratio with latest values")
    else:
        details.append("Insufficient cash or revenue data")
    
    # 4. Consistency in share count - Munger prefers stable/decreasing shares
    market_data = fundamental_data.get("market_data", {})
    shareholding = fundamental_data.get("shareholding_pattern", {})
    
    # First try to get historical share count data
    historical_shares = None
    if "No. of Shares" in shareholding:
        historical_shares = shareholding["No. of Shares"]
        historical_shares = [float(s) if s is not None else None for s in historical_shares]
    
    # If no historical data, use current shares only
    current_shares = None
    if "No. of Shares" in market_data:
        try:
            current_shares = float(market_data["No. of Shares"])
        except (ValueError, TypeError):
            pass
    
    if historical_shares and len(historical_shares) >= 3:
        # Remove None values
        historical_shares = [s for s in historical_shares if s is not None]
        
        if len(historical_shares) >= 3:
            if historical_shares[0] < historical_shares[-1] * 0.95:  # 5%+ reduction in shares
                score += 2
                details.append("Shareholder-friendly: Reducing share count over time")
            elif historical_shares[0] < historical_shares[-1] * 1.05:  # Stable share count
                score += 1
                details.append("Stable share count: Limited dilution")
            elif historical_shares[0] > historical_shares[-1] * 1.2:  # >20% dilution
                score -= 1  # Penalty for excessive dilution
                details.append("Concerning dilution: Share count increased significantly")
            else:
                details.append("Moderate share count increase over time")
        else:
            details.append("Insufficient historical share count data after filtering")
    elif current_shares:
        details.append("Only current share count data available - cannot assess dilution")
    else:
        details.append("Share count data not available")
    
    # 5. Dividend consistency - Munger respects disciplined capital return
    profit_loss = fundamental_data.get("profit_loss", {})
    dividend_values = []
    
    for div_key in ["Dividend Payout %", "Dividend %"]:
        if div_key in profit_loss:
            dividend_values = profit_loss[div_key]
            dividend_values = [float(d) if d is not None else None for d in dividend_values]
            break
    
    if dividend_values and len(dividend_values) >= 3:
        # Remove None values
        dividend_values = [d for d in dividend_values if d is not None]
        
        if len(dividend_values) >= 3:
            # Count consistent dividend periods
            consistent_periods = sum(1 for i in range(1, len(dividend_values)) 
                               if dividend_values[i] > 0 and abs(dividend_values[i] - dividend_values[i-1]) / max(0.01, dividend_values[i-1]) < 0.2)
            
            if consistent_periods >= len(dividend_values) - 1:
                score += 2
                details.append("Disciplined capital return: Consistent dividend policy")
            elif sum(1 for d in dividend_values if d > 0) >= len(dividend_values) * 0.8:
                score += 1
                details.append("Regular dividend payments, though somewhat variable")
            else:
                details.append("Irregular dividend policy")
        else:
            details.append("Insufficient dividend data after filtering")
    else:
        details.append("Dividend data not available or insufficient")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 12 (3+3+2+2+2)
    # Minimum possible score could be -1 due to penalty
    final_score = max(0, min(10, (score + 1) * 10 / 13))
    
    # Determine rating based on score
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Management Quality: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Exceptional capital allocation and business stewardship."
    elif final_score >= 5:
        summary += ". Solid management practices with reasonable capital allocation."
    elif final_score >= 2.5:
        summary += ". Mixed management quality with some concerning practices."
    else:
        summary += ". Poor management decisions and capital allocation."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

def analyze_predictability(ticker: str) -> dict:
    """
    Assess the predictability of the business - Munger strongly prefers businesses
    whose future operations and cashflows are relatively easy to predict.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"score": 0, "details": "Insufficient data to analyze business predictability", "rating": "Poor", "summary": "Business Predictability: Poor (0/10 points). Insufficient data."}
    
    # 1. Revenue stability and growth
    profit_loss = fundamental_data.get("profit_loss", {})
    revenues = []
    
    if "Revenue" in profit_loss:
        revenues = profit_loss["Revenue"]
        revenues = [float(r) if r is not None else None for r in revenues]
        # Remove None values
        revenues = [r for r in revenues if r is not None]
    
    if revenues and len(revenues) >= 5:
        # Calculate year-over-year growth rates
        growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
        
        avg_growth = sum(growth_rates) / len(growth_rates)
        growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
        
        if avg_growth > 0.05 and growth_volatility < 0.1:
            # Steady, consistent growth (Munger loves this)
            score += 3
            details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
        elif avg_growth > 0 and growth_volatility < 0.2:
            # Positive but somewhat volatile growth
            score += 2
            details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth with some volatility")
        elif avg_growth > 0:
            # Growing but unpredictable
            score += 1
            details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth with high volatility")
        else:
            details.append(f"Declining or highly unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        details.append("Insufficient revenue history for predictability analysis")
    
    # 2. Operating income stability
    operating_income = []
    
    if "Operating Profit" in profit_loss:
        operating_income = profit_loss["Operating Profit"]
        operating_income = [float(oi) if oi is not None else None for oi in operating_income]
        # Remove None values
        operating_income = [oi for oi in operating_income if oi is not None]
    
    if operating_income and len(operating_income) >= 5:
        # Count positive operating income periods
        positive_periods = sum(1 for income in operating_income if income > 0)
        
        if positive_periods == len(operating_income):
            # Consistently profitable operations
            score += 3
            details.append("Highly predictable operations: Operating income positive in all periods")
        elif positive_periods >= len(operating_income) * 0.8:
            # Mostly profitable operations
            score += 2
            details.append(f"Predictable operations: Operating income positive in {positive_periods}/{len(operating_income)} periods")
        elif positive_periods >= len(operating_income) * 0.6:
            # Somewhat profitable operations
            score += 1
            details.append(f"Somewhat predictable operations: Operating income positive in {positive_periods}/{len(operating_income)} periods")
        else:
            details.append(f"Unpredictable operations: Operating income positive in only {positive_periods}/{len(operating_income)} periods")
    else:
        details.append("Insufficient operating income history")
    
    # 3. Margin consistency - Munger values stable margins
    operating_margin_data = get_operating_margin(ticker)
    
    if operating_margin_data and "historical_margins" in operating_margin_data:
        op_margins = operating_margin_data["historical_margins"]
        # Filter out None values
        op_margins = [m for m in op_margins if m is not None]
        
        if op_margins and len(op_margins) >= 5:
            # Calculate margin volatility
            avg_margin = sum(op_margins) / len(op_margins)
            margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
            
            if margin_volatility < 0.03:  # Very stable margins
                score += 2
                details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
            elif margin_volatility < 0.07:  # Moderately stable margins
                score += 1
                details.append(f"Moderately predictable margins: {avg_margin:.1%} avg with some volatility")
            else:
                details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility ({margin_volatility:.1%})")
        else:
            details.append("Insufficient margin history after filtering")
    else:
        details.append("No operating margin data available")
    
    # 4. Cash generation reliability
    fcf_data = get_free_cash_flow(ticker)
    
    if fcf_data and "historical_fcf" in fcf_data:
        fcf_values = fcf_data["historical_fcf"]
        # Filter out None values
        fcf_values = [fcf for fcf in fcf_values if fcf is not None]
        
        if fcf_values and len(fcf_values) >= 5:
            # Count positive FCF periods
            positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
            
            if positive_fcf_periods == len(fcf_values):
                # Consistently positive FCF
                score += 2
                details.append("Highly predictable cash generation: Positive FCF in all periods")
            elif positive_fcf_periods >= len(fcf_values) * 0.8:
                # Mostly positive FCF
                score += 1
                details.append(f"Predictable cash generation: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
            else:
                details.append(f"Unpredictable cash generation: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append("Insufficient free cash flow history after filtering")
    else:
        details.append("Free cash flow data not available")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (3+3+2+2)
    final_score = min(10, score * 10 / 10)
    
    # Determine rating based on score
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Business Predictability: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Highly consistent and predictable business operations."
    elif final_score >= 5:
        summary += ". Good predictability with some consistency in results."
    elif final_score >= 2.5:
        summary += ". Moderate predictability with some volatility."
    else:
        summary += ". Unpredictable business with erratic results."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

def calculate_munger_valuation(ticker: str, market_cap: Optional[float] = None) -> dict:
    """
    Calculate intrinsic value using Munger's approach:
    - Focus on owner earnings (approximated by FCF)
    - Simple multiple on normalized earnings
    - Prefer paying a fair price for a wonderful business
    """
    score = 0
    details = []
    
    # Get market cap if not provided
    if market_cap is None:
        market_cap_data = get_market_cap(ticker)
        if market_cap_data:
            market_cap = market_cap_data
    
    if not market_cap or market_cap <= 0:
        details.append("Market cap data not available or invalid.")
        return {"score": 0, "details": "; ".join(details), "rating": "Unknown", "summary": "Valuation: Unknown (0/10 points). Insufficient data."}
    
    # Get free cash flow data
    fcf_data = get_free_cash_flow(ticker)
    
    if not fcf_data or "historical_fcf" not in fcf_data or not fcf_data["historical_fcf"]:
        return {
            "score": 0,
            "details": "Insufficient free cash flow data for valuation",
            "rating": "Unknown",
            "summary": "Valuation: Unknown (0/10 points). Insufficient free cash flow data."
        }
    
    fcf_values = fcf_data["historical_fcf"]
    # Filter out None values
    fcf_values = [fcf for fcf in fcf_values if fcf is not None]
    
    if not fcf_values or len(fcf_values) < 3:
        return {
            "score": 0,
            "details": "Insufficient valid free cash flow data for valuation",
            "rating": "Unknown",
            "summary": "Valuation: Unknown (0/10 points). Insufficient valid free cash flow data."
        }
    
    # 1. Normalize earnings by taking average of last 3-5 years
    # (Munger prefers to normalize earnings to avoid over/under-valuation based on cyclical factors)
    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    
    if normalized_fcf <= 0:
        return {
            "score": 0,
            "details": f"Negative or zero normalized FCF ({normalized_fcf}), cannot value",
            "intrinsic_value": None,
            "rating": "Unknown",
            "summary": "Valuation: Unknown (0/10 points). Negative or zero normalized free cash flow."
        }
    
    # 2. Calculate FCF yield (inverse of P/FCF multiple)
    fcf_yield = normalized_fcf / market_cap
    
    # 3. Apply Munger's FCF multiple based on business quality
    # Munger would pay higher multiples for wonderful businesses
    # Let's use a sliding scale where higher FCF yields are more attractive
    if fcf_yield > 0.08:  # >8% FCF yield (P/FCF < 12.5x)
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:  # >5% FCF yield (P/FCF < 20x)
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:  # >3% FCF yield (P/FCF < 33x)
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")
    
    # 4. Calculate simple intrinsic value range
    # Munger tends to use straightforward valuations, avoiding complex DCF models
    conservative_value = normalized_fcf * 10  # 10x FCF = 10% yield
    reasonable_value = normalized_fcf * 15    # 15x FCF ≈ 6.7% yield
    optimistic_value = normalized_fcf * 20    # 20x FCF = 5% yield
    
    # 5. Calculate margins of safety
    current_to_reasonable = (reasonable_value - market_cap) / market_cap
    
    if current_to_reasonable > 0.3:  # >30% upside
        score += 3
        details.append(f"Large margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > 0.1:  # >10% upside
        score += 2
        details.append(f"Moderate margin of safety: {current_to_reasonable:.1%} upside to reasonable value")
    elif current_to_reasonable > -0.1:  # Within 10% of reasonable value
        score += 1
        details.append(f"Fair price: Within 10% of reasonable value ({current_to_reasonable:.1%})")
    else:
        details.append(f"Expensive: {-current_to_reasonable:.1%} premium to reasonable value")
    
    # 6. Check earnings trajectory for additional context
    # Munger likes growing owner earnings
    if len(fcf_values) >= 3:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
        
        if recent_avg > older_avg * 1.2:  # >20% growth in FCF
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")
    
    # Scale score to 0-10 range
    # Maximum possible raw score would be 10 (4+3+3)
    final_score = min(10, score * 10 / 10)
    
    # Determine rating based on score
    if final_score >= 7.5:
        rating = "Undervalued"
    elif final_score >= 5:
        rating = "Fairly Valued"
    elif final_score >= 2.5:
        rating = "Slightly Overvalued"
    else:
        rating = "Overvalued"
    
    # Generate summary
    summary = f"Valuation: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += f". Significant undervaluation with {current_to_reasonable:.1%} upside potential."
    elif final_score >= 5:
        summary += f". Reasonable valuation with modest upside potential."
    elif final_score >= 2.5:
        summary += f". Slightly expensive relative to intrinsic value."
    else:
        summary += f". Significantly overvalued with limited margin of safety."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary,
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value
        },
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf
    } 

def analyze_news_sentiment(news_items: list) -> str:
    """
    Simple qualitative analysis of recent news.
    Munger pays attention to significant news but doesn't overreact to short-term stories.
    """
    if not news_items or len(news_items) == 0:
        return "No news data available"
    
    # Just return a simple count for now - in a real implementation, this would use NLP
    return f"Qualitative review of {len(news_items)} recent news items would be needed"


class CharlieMungerAgnoAgent():
    """Agno-based agent implementing Charlie Munger's investment principles."""
    
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

    def _prepare_agent_prompt(self, ticker: str, analysis_summary: Dict[str, Any]) -> str:
        """
        Prepare the prompt for the LLM based on the calculated analysis.
        Uses the prompt structure from the LangChain version.
        """
        prompt_template = """You are a Charlie Munger AI agent, making investment decisions using his principles:

            1. Focus on the quality and predictability of the business.
            2. Rely on mental models from multiple disciplines to analyze investments.
            3. Look for strong, durable competitive advantages (moats).
            4. Emphasize long-term thinking and patience.
            5. Value management integrity and competence.
            6. Prioritize businesses with high returns on invested capital.
            7. Pay a fair price for wonderful businesses.
            8. Never overpay, always demand a margin of safety.
            9. Avoid complexity and businesses you don't understand.
            10. "Invert, always invert" - focus on avoiding stupidity rather than seeking brilliance.
            
            Please analyze {ticker} using the following data:

            Moat Strength Analysis:
            {moat_analysis}

            Management Quality Analysis:
            {management_analysis}

            Business Predictability Analysis:
            {predictability_analysis}

            Valuation Analysis:
            {valuation_analysis}

            Based on this analysis and Charlie Munger's investment philosophy, would you recommend investing in {ticker}?
            Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
            Structure your reasoning to follow Munger's mental models approach.
            """
        return prompt_template.format(
            ticker=ticker,
            moat_analysis=json.dumps(analysis_summary["moat_analysis"], indent=2),
            management_analysis=json.dumps(analysis_summary["management_analysis"], indent=2),
            predictability_analysis=json.dumps(analysis_summary["predictability_analysis"], indent=2),
            valuation_analysis=json.dumps(analysis_summary["valuation_analysis"], indent=2)
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Charlie Munger's principles.
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "charlie_munger_agno_agent"

        data = state.get("data", {})
        metadata = state.get("metadata", {})
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
                print(f"Analyzing {ticker}...")
                
                # Check for presence of key data sections
                fundamental_data = fetch_fundamental_data(ticker)
                if not fundamental_data:
                    results[ticker] = {"error": f"No fundamental data found for {ticker}"}
                    continue
                
                has_profit_loss = "profit_loss" in fundamental_data
                has_balance_sheet = "balance_sheet" in fundamental_data
                has_cash_flows = "cash_flows" in fundamental_data
                has_market_data = "market_data" in fundamental_data
                
                # Debug info about available data
                data_availability = {
                    "has_profit_loss": has_profit_loss,
                    "has_balance_sheet": has_balance_sheet,
                    "has_cash_flows": has_cash_flows,
                    "has_market_data": has_market_data
                }
                print(f"Data availability for {ticker}: {data_availability}")
                
                # Perform analysis using the direct API
                moat_analysis = analyze_moat_strength(ticker)
                management_analysis = analyze_management_quality(ticker)
                predictability_analysis = analyze_predictability(ticker)
                valuation_analysis = calculate_munger_valuation(ticker)
                
                # Combine scores with Munger's weighting preferences
                # Munger weights quality and predictability higher than current valuation
                total_score = (
                    moat_analysis.get("score", 0) * 0.35 +
                    management_analysis.get("score", 0) * 0.25 +
                    predictability_analysis.get("score", 0) * 0.25 +
                    valuation_analysis.get("score", 0) * 0.15
                )
                
                # Determine signal based on Munger's standards
                if total_score >= 7.5:  # Munger has very high standards
                    signal = "bullish"
                elif total_score <= 4.5:
                    signal = "bearish"
                else:
                    signal = "neutral"
                
                analysis = {
                    "moat_analysis": moat_analysis,
                    "management_analysis": management_analysis,
                    "predictability_analysis": predictability_analysis,
                    "valuation_analysis": valuation_analysis,
                    "signal": signal,
                    "score": total_score,
                    "max_score": 10.0
                }
                
                # Debug print the analysis
                print(f"Analysis for {ticker}:")
                print(json.dumps(analysis, indent=2))
                
                # Prepare prompt and run LLM
                prompt = self._prepare_agent_prompt(ticker, analysis)
                llm_response = self.agent.run(prompt)
                
                # Parse LLM response into structured signal
                signal = self._parse_response(llm_response, ticker)
                
                # Store results
                results[ticker] = {
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                    "analysis": analysis,
                    "data_availability": data_availability
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {"error": f"Error analyzing {ticker}: {str(e)}"}
        
        return {agent_name: results}
    
    def _parse_response(self, response: str, ticker: str) -> CharlieMungerSignal:
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
            
            # Truncate reasoning to reasonable length
            reasoning = response_text[:1000] if len(response_text) > 1000 else response_text
            
            return CharlieMungerSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
            return CharlieMungerSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )

    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Charlie Munger's investment criteria"""
        print(f"Starting Charlie Munger-based analysis for {ticker}")
        
        # Analyze the four key aspects of the investment
        moat_analysis = analyze_moat_strength(ticker)
        management_analysis = analyze_management_quality(ticker)
        predictability_analysis = analyze_predictability(ticker)
        valuation_analysis = calculate_munger_valuation(ticker)
        
        # Calculate overall score with Munger's weighting preferences
        total_score = (
            moat_analysis.get("score", 0) * 0.35 +
            management_analysis.get("score", 0) * 0.25 +
            predictability_analysis.get("score", 0) * 0.25 +
            valuation_analysis.get("score", 0) * 0.15
        )
        
        # Determine investment recommendation
        if total_score >= 7.5:  # Munger has very high standards
            recommendation = "Strong Buy"
        elif total_score >= 6.0:
            recommendation = "Buy"
        elif total_score >= 5.0:
            recommendation = "Hold"
        elif total_score >= 3.5:
            recommendation = "Monitor"
        else:
            recommendation = "Avoid"
        
        # Generate comprehensive report
        report = {
            "ticker": ticker,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            "overall_score": total_score,
            "recommendation": recommendation,
            "summary": f"""
Charlie Munger Analysis for {ticker}:

{moat_analysis['summary']}

{management_analysis['summary']}

{predictability_analysis['summary']}

{valuation_analysis['summary']}

Overall Score: {total_score:.1f}/10
Recommendation: {recommendation}
"""
        }
        
        print(f"Completed Charlie Munger-based analysis for {ticker}")
        return report


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_state = {
        "data": {
            "tickers": ["HDFCBANK"],  # Example ticker
            "end_date": "2023-12-31"  # Optional end date
        }
        # Optional metadata for model selection can be added here
    }
    try:
        agent = CharlieMungerAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 