"""
Phil Fisher Investing Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
import math
import re
import statistics
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
from tools.ohlcv import fetch_ohlcv_data

# Pydantic model for the output signal
class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str 

def analyze_growth_quality(ticker: str) -> Dict[str, Any]:
    """
    Evaluate growth & quality:
    - Consistent Revenue Growth
    - Consistent EPS Growth
    - R&D as a % of Revenue (if relevant, indicative of future-oriented spending)
    
    Phil Fisher places significant emphasis on companies with sustainable growth
    and investment in research and development.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient financial data for growth/quality analysis",
            "rating": "Poor",
            "summary": "Growth & Quality: Poor (0/10 points). Insufficient data."
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    
    # 1. Analyze Revenue Growth (YoY)
    revenue_values = []
    if "Revenue" in profit_loss:
        revenue_values = profit_loss["Revenue"]
        revenue_values = [float(r) if r is not None else None for r in revenue_values]
    
    if revenue_values and len([r for r in revenue_values if r is not None]) >= 2:
        # Remove None values
        valid_revenues = [r for r in revenue_values if r is not None]
        
        # We'll look at the earliest vs. latest to gauge multi-year growth
        latest_rev = valid_revenues[0]  # Most recent first
        oldest_rev = valid_revenues[-1]
        
        if oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / abs(oldest_rev)
            
            if rev_growth > 0.80:  # 80%+ growth over the period
                score += 3
                details.append(f"Very strong multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.40:  # 40%+ growth
                score += 2
                details.append(f"Moderate multi-period revenue growth: {rev_growth:.1%}")
            elif rev_growth > 0.10:  # 10%+ growth
                score += 1
                details.append(f"Slight multi-period revenue growth: {rev_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period revenue growth: {rev_growth:.1%}")
        else:
            details.append("Oldest revenue is zero/negative; cannot compute growth.")
    else:
        details.append("Not enough revenue data points for growth calculation.")
    
    # 2. Analyze EPS Growth (YoY)
    eps_values = []
    for eps_key in ["EPS", "Earnings Per Share", "Diluted EPS"]:
        if eps_key in profit_loss:
            eps_values = profit_loss[eps_key]
            eps_values = [float(e) if e is not None else None for e in eps_values]
            break
    
    if eps_values and len([e for e in eps_values if e is not None]) >= 2:
        # Remove None values
        valid_eps = [e for e in eps_values if e is not None]
        
        latest_eps = valid_eps[0]
        oldest_eps = valid_eps[-1]
        
        if abs(oldest_eps) > 1e-9:  # Ensure denominator isn't too close to zero
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            
            if eps_growth > 0.80:
                score += 3
                details.append(f"Very strong multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.40:
                score += 2
                details.append(f"Moderate multi-period EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                score += 1
                details.append(f"Slight multi-period EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative multi-period EPS growth: {eps_growth:.1%}")
        else:
            details.append("Oldest EPS near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data points for growth calculation.")
    
    # 3. Analyze R&D as % of Revenue (if available)
    rnd_values = []
    for rnd_key in ["Research and Development", "R&D Expense", "Research & Development"]:
        if rnd_key in profit_loss:
            rnd_values = profit_loss[rnd_key]
            rnd_values = [float(r) if r is not None else None for r in rnd_values]
            break
    
    if rnd_values and revenue_values and min(len(rnd_values), len(revenue_values)) > 0:
        # Use the most recent period for R&D analysis
        min_length = min(len(rnd_values), len(revenue_values))
        valid_rnd = [r for r in rnd_values[:min_length] if r is not None]
        valid_rev = [r for r in revenue_values[:min_length] if r is not None]
        
        if valid_rnd and valid_rev:
            recent_rnd = valid_rnd[0]
            recent_rev = valid_rev[0]
            
            if recent_rev > 0:
                rnd_ratio = recent_rnd / recent_rev
                
                # Fisher admired companies that invest aggressively in R&D
                # But the appropriate level depends on the industry
                if 0.03 <= rnd_ratio <= 0.15:  # 3-15% is healthy for most industries
                    score += 3
                    details.append(f"R&D ratio {rnd_ratio:.1%} indicates significant investment in future growth")
                elif rnd_ratio > 0.15:  # Very high R&D
                    score += 2
                    details.append(f"R&D ratio {rnd_ratio:.1%} is very high (could be good if well-managed)")
                elif rnd_ratio > 0.0:  # Some R&D
                    score += 1
                    details.append(f"R&D ratio {rnd_ratio:.1%} is somewhat low but still positive")
                else:
                    details.append("No meaningful R&D expense ratio")
            else:
                details.append("Revenue is zero or negative; cannot compute R&D ratio.")
        else:
            details.append("No valid R&D or revenue data for recent period.")
    else:
        details.append("Insufficient R&D data to evaluate")
    
    # Adjust score to 0-10 scale
    max_raw_score = 9  # Maximum possible raw score (3+3+3)
    final_score = min(10, score * 10 / max_raw_score)
    
    # Determine rating
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Growth & Quality: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Exceptional growth with strong R&D investment."
    elif final_score >= 5:
        summary += ". Good growth profile with reasonable R&D investment."
    elif final_score >= 2.5:
        summary += ". Average growth with limited R&D investment."
    else:
        summary += ". Weak growth profile with inadequate R&D investment."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

def analyze_margins_stability(ticker: str) -> Dict[str, Any]:
    """
    Analyze margin consistency (gross/operating margin) and stability over time.
    
    Phil Fisher emphasizes the importance of consistent and high margins as 
    indicators of a company's competitive advantage and management quality.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for margin stability analysis",
            "rating": "Poor",
            "summary": "Margins Stability: Poor (0/10 points). Insufficient data."
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    
    # 1. Operating Margin Consistency
    operating_margins = []
    
    # Try to get operating margin from our utility function first
    margin_data = get_operating_margin(ticker)
    if margin_data and "historical_margins" in margin_data and margin_data["historical_margins"]:
        operating_margins = margin_data["historical_margins"]
    else:
        # If not available from utility, try to get from profit_loss data
        for margin_key in ["Operating Margin", "Operating Margin %", "OPM %"]:
            if margin_key in profit_loss:
                operating_margins = profit_loss[margin_key]
                # Convert from percentage to decimal if needed
                operating_margins = [float(m)/100 if m is not None and float(m) > 1 else float(m) if m is not None else None for m in operating_margins]
                break
    
    if operating_margins and len([m for m in operating_margins if m is not None]) >= 2:
        # Remove None values
        valid_margins = [m for m in operating_margins if m is not None]
        
        # Check if margins are stable or improving
        oldest_margin = valid_margins[-1]
        newest_margin = valid_margins[0]
        
        if newest_margin >= oldest_margin > 0:
            score += 2
            details.append(f"Operating margin stable or improving ({oldest_margin:.1%} → {newest_margin:.1%})")
        elif newest_margin > 0:
            score += 1
            details.append(f"Operating margin positive but slightly declined ({oldest_margin:.1%} → {newest_margin:.1%})")
        else:
            details.append(f"Operating margin negative or severely declined ({oldest_margin:.1%} → {newest_margin:.1%})")
    else:
        details.append("Not enough operating margin data points")
    
    # 2. Gross Margin Level
    gross_margins = []
    for margin_key in ["Gross Margin", "Gross Margin %", "GPM %"]:
        if margin_key in profit_loss:
            gross_margins = profit_loss[margin_key]
            # Convert from percentage to decimal if needed
            gross_margins = [float(m)/100 if m is not None and float(m) > 1 else float(m) if m is not None else None for m in gross_margins]
            break
    
    if gross_margins and len([m for m in gross_margins if m is not None]) >= 1:
        # Use the most recent value
        valid_margins = [m for m in gross_margins if m is not None]
        recent_gm = valid_margins[0]
        
        if recent_gm > 0.5:  # 50%+ is excellent
            score += 2
            details.append(f"Strong gross margin: {recent_gm:.1%}")
        elif recent_gm > 0.3:  # 30%+ is good
            score += 1
            details.append(f"Moderate gross margin: {recent_gm:.1%}")
        else:
            details.append(f"Low gross margin: {recent_gm:.1%}")
    else:
        details.append("No gross margin data available")
    
    # 3. Multi-year Margin Stability
    if operating_margins and len([m for m in operating_margins if m is not None]) >= 3:
        # Remove None values
        valid_margins = [m for m in operating_margins if m is not None]
        
        try:
            # Calculate standard deviation to measure volatility
            stdev = statistics.pstdev(valid_margins)
            
            # Calculate coefficient of variation if possible
            mean_margin = statistics.mean(valid_margins)
            if mean_margin > 0:
                cv = stdev / mean_margin
                
                if cv < 0.1:  # Very stable margins
                    score += 2
                    details.append(f"Operating margin extremely stable over multiple years (CV: {cv:.2f})")
                elif cv < 0.2:  # Reasonably stable
                    score += 1
                    details.append(f"Operating margin reasonably stable (CV: {cv:.2f})")
                else:
                    details.append(f"Operating margin volatility is high (CV: {cv:.2f})")
            else:
                details.append("Mean operating margin is zero or negative; stability analysis skipped")
        except Exception as e:
            details.append(f"Error calculating margin stability: {str(e)}")
    else:
        details.append("Not enough margin data points for volatility check")
    
    # Adjust score to 0-10 scale
    max_raw_score = 6  # Maximum possible raw score (2+2+2)
    final_score = min(10, score * 10 / max_raw_score)
    
    # Determine rating
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Margins Stability: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Strong and consistent margins indicating competitive advantage."
    elif final_score >= 5:
        summary += ". Good margin profile with reasonable stability."
    elif final_score >= 2.5:
        summary += ". Average margins with some volatility."
    else:
        summary += ". Weak or volatile margins indicating competitive pressure."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

def analyze_management_efficiency(ticker: str) -> Dict[str, Any]:
    """
    Evaluate management efficiency & leverage:
    - Return on Equity (ROE)
    - Debt-to-Equity ratio
    - Free Cash Flow consistency
    
    Phil Fisher placed significant emphasis on management quality,
    focusing on their capital allocation decisions and efficiency.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for management efficiency analysis",
            "rating": "Poor",
            "summary": "Management Efficiency: Poor (0/10 points). Insufficient data.",
            "max_score": 10
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    balance_sheet = fundamental_data.get("balance_sheet", {})
    cash_flows = fundamental_data.get("cash_flows", {})
    
    # 1. Return on Equity (ROE)
    # Try to get from our utility function first
    roe_data = get_roe(ticker)
    if roe_data and "current_roe" in roe_data and roe_data["current_roe"] is not None:
        current_roe = roe_data["current_roe"]
        
        if current_roe > 0.2:  # 20%+ ROE is excellent
            score += 3
            details.append(f"High ROE: {current_roe:.1%}")
        elif current_roe > 0.15:  # 15%+ ROE is good
            score += 2
            details.append(f"Good ROE: {current_roe:.1%}")
        elif current_roe > 0.10:  # 10%+ ROE is reasonable
            score += 1
            details.append(f"Acceptable ROE: {current_roe:.1%}")
        elif current_roe > 0:
            details.append(f"Low ROE: {current_roe:.1%}")
        else:
            details.append(f"Negative ROE: {current_roe:.1%}")
    else:
        # If not available from utility, try to calculate from financial data
        net_income_values = []
        equity_values = []
        
        # Get net income values
        for income_key in ["Net Income", "Net Profit", "Profit After Tax"]:
            if income_key in profit_loss:
                net_income_values = profit_loss[income_key]
                net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
                break
        
        # Get equity values
        for equity_key in ["Total Equity", "Shareholders Equity", "Total Shareholders Equity"]:
            if equity_key in balance_sheet:
                equity_values = balance_sheet[equity_key]
                equity_values = [float(eq) if eq is not None else None for eq in equity_values]
                break
        
        if net_income_values and equity_values and len(net_income_values) > 0 and len(equity_values) > 0:
            # Use the most recent values
            recent_ni = net_income_values[0]
            recent_eq = equity_values[0]
            
            if recent_eq and recent_eq > 0:
                roe = recent_ni / recent_eq
                
                if roe > 0.2:
                    score += 3
                    details.append(f"High ROE: {roe:.1%}")
                elif roe > 0.15:
                    score += 2
                    details.append(f"Good ROE: {roe:.1%}")
                elif roe > 0.10:
                    score += 1
                    details.append(f"Acceptable ROE: {roe:.1%}")
                elif roe > 0:
                    details.append(f"Low ROE: {roe:.1%}")
                else:
                    details.append(f"Negative ROE: {roe:.1%}")
            else:
                details.append("Equity value is zero or negative; ROE calculation skipped")
        else:
            details.append("Insufficient data for ROE calculation")
    
    # 2. Debt-to-Equity ratio
    # Try to get from our utility function first
    debt_equity_data = get_debt_to_equity(ticker)
    if debt_equity_data and "current_ratio" in debt_equity_data and debt_equity_data["current_ratio"] is not None:
        debt_to_equity = debt_equity_data["current_ratio"]
        
        if debt_to_equity < 0.3:  # Very low debt
            score += 2
            details.append(f"Low debt-to-equity: {debt_to_equity:.2f}")
        elif debt_to_equity < 1.0:  # Reasonable debt
            score += 1
            details.append(f"Manageable debt-to-equity: {debt_to_equity:.2f}")
        else:
            details.append(f"High debt-to-equity: {debt_to_equity:.2f}")
    else:
        # If not available from utility, try to calculate from financial data
        debt_values = []
        
        # Get debt values
        for debt_key in ["Total Debt", "Long Term Debt", "Short Term Debt"]:
            if debt_key in balance_sheet:
                debt_values = balance_sheet[debt_key]
                debt_values = [float(d) if d is not None else None for d in debt_values]
                break
        
        if debt_values and equity_values and len(debt_values) > 0 and len(equity_values) > 0:
            # Use the most recent values
            recent_debt = debt_values[0]
            recent_equity = equity_values[0]
            
            if recent_equity and recent_equity > 0:
                dte = recent_debt / recent_equity
                
                if dte < 0.3:
                    score += 2
                    details.append(f"Low debt-to-equity: {dte:.2f}")
                elif dte < 1.0:
                    score += 1
                    details.append(f"Manageable debt-to-equity: {dte:.2f}")
                else:
                    details.append(f"High debt-to-equity: {dte:.2f}")
            else:
                details.append("Equity value is zero or negative; D/E calculation skipped")
        else:
            details.append("Insufficient data for debt-to-equity calculation")
    
    # 3. Free Cash Flow Consistency
    # Try to get from our utility function first
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "historical_fcf" in fcf_data and fcf_data["historical_fcf"]:
        historical_fcf = fcf_data["historical_fcf"]
        
        if len(historical_fcf) >= 2:
            # Check if FCF is positive in recent years
            positive_fcf_count = sum(1 for fcf in historical_fcf if fcf > 0)
            fcf_ratio = positive_fcf_count / len(historical_fcf)
            
            if fcf_ratio >= 0.8:  # 80%+ of periods have positive FCF
                score += 1
                details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(historical_fcf)})")
            elif fcf_ratio >= 0.5:  # At least half have positive FCF
                score += 0.5
                details.append(f"Mixed FCF results ({positive_fcf_count}/{len(historical_fcf)} positive)")
            else:
                details.append(f"Free cash flow is inconsistent or often negative ({positive_fcf_count}/{len(historical_fcf)} positive)")
        else:
            details.append("Insufficient historical FCF data for consistency check")
    else:
        # If not available from utility, try to get from cash_flows data
        fcf_values = []
        
        # Get FCF values
        for fcf_key in ["Free Cash Flow", "FCF"]:
            if fcf_key in cash_flows:
                fcf_values = cash_flows[fcf_key]
                fcf_values = [float(fcf) if fcf is not None else None for fcf in fcf_values]
                break
        
        if fcf_values and len([f for f in fcf_values if f is not None]) >= 2:
            # Remove None values
            valid_fcf = [f for f in fcf_values if f is not None]
            
            # Check if FCF is positive in recent years
            positive_fcf_count = sum(1 for fcf in valid_fcf if fcf > 0)
            fcf_ratio = positive_fcf_count / len(valid_fcf)
            
            if fcf_ratio >= 0.8:
                score += 1
                details.append(f"Majority of periods have positive FCF ({positive_fcf_count}/{len(valid_fcf)})")
            elif fcf_ratio >= 0.5:
                score += 0.5
                details.append(f"Mixed FCF results ({positive_fcf_count}/{len(valid_fcf)} positive)")
            else:
                details.append(f"Free cash flow is inconsistent or often negative ({positive_fcf_count}/{len(valid_fcf)} positive)")
        else:
            details.append("Insufficient FCF data for consistency check")
    
    # Normalize score to 0-10 scale
    max_raw_score = 6  # Maximum possible raw score (3+2+1)
    final_score = min(10, score * 10 / max_raw_score)
    
    # Determine rating
    if final_score >= 7.5:
        rating = "Excellent"
    elif final_score >= 5:
        rating = "Good"
    elif final_score >= 2.5:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    summary = f"Management Efficiency: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Exceptional capital allocation with efficient use of resources."
    elif final_score >= 5:
        summary += ". Good management with reasonable return on equity."
    elif final_score >= 2.5:
        summary += ". Average management efficiency with some concerns."
    else:
        summary += ". Poor capital allocation and inefficient use of resources."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary,
        "max_score": 10
    } 

def analyze_valuation(ticker: str, market_cap: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze valuation using Phil Fisher's approach:
    - P/E ratio
    - P/FCF ratio
    - Enterprise Value metrics (optional)
    
    Fisher was willing to pay a premium for quality growth companies,
    but still wanted reasonable valuations relative to earnings power.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data to perform valuation",
            "rating": "Unknown",
            "summary": "Valuation: Unknown (0/10 points). Insufficient data."
        }
    
    # Get market cap if not provided
    if market_cap is None:
        market_cap_data = get_market_cap(ticker)
        if isinstance(market_cap_data, dict) and "current_market_cap" in market_cap_data:
            market_cap = market_cap_data["current_market_cap"]
        elif isinstance(market_cap_data, (int, float)):
            market_cap = market_cap_data
        else:
            return {
                "score": 0,
                "details": "Market cap data not available",
                "rating": "Unknown",
                "summary": "Valuation: Unknown (0/10 points). Insufficient market data."
            }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    
    # 1. Analyze P/E Ratio
    # Try to get from our utility function first
    pe_data = get_pe_ratio(ticker)
    if pe_data and "current_pe" in pe_data and pe_data["current_pe"] is not None:
        pe_ratio = pe_data["current_pe"]
        
        # Fisher would accept higher P/Es for quality growth companies
        if pe_ratio < 20:  # Attractive valuation
            score += 2
            details.append(f"Reasonably attractive P/E: {pe_ratio:.2f}")
        elif pe_ratio < 30:  # Acceptable for a growth company
            score += 1
            details.append(f"Somewhat high but possibly justifiable P/E: {pe_ratio:.2f}")
        else:
            details.append(f"Very high P/E: {pe_ratio:.2f}")
    else:
        # If not available from utility, try to calculate from financial data
        net_income_values = []
        
        # Get net income values
        for income_key in ["Net Income", "Net Profit", "Profit After Tax"]:
            if income_key in profit_loss:
                net_income_values = profit_loss[income_key]
                net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
                break
        
        if net_income_values and len(net_income_values) > 0 and net_income_values[0] is not None:
            recent_net_income = net_income_values[0]
            
            if recent_net_income > 0 and market_cap is not None:
                pe = market_cap / recent_net_income
                
                if pe < 20:
                    score += 2
                    details.append(f"Reasonably attractive P/E: {pe:.2f}")
                elif pe < 30:
                    score += 1
                    details.append(f"Somewhat high but possibly justifiable P/E: {pe:.2f}")
                else:
                    details.append(f"Very high P/E: {pe:.2f}")
            else:
                details.append("Net income is zero or negative; P/E calculation skipped")
        else:
            details.append("No valid net income data for P/E calculation")
    
    # 2. Analyze P/FCF Ratio
    # Try to get from our utility function first
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "current_fcf" in fcf_data and fcf_data["current_fcf"] is not None:
        recent_fcf = fcf_data["current_fcf"]
        
        if recent_fcf > 0 and market_cap is not None:
            pfcf = market_cap / recent_fcf
            
            if pfcf < 20:  # Attractive valuation
                score += 2
                details.append(f"Reasonable P/FCF: {pfcf:.2f}")
            elif pfcf < 30:  # Acceptable for a growth company
                score += 1
                details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
            else:
                details.append(f"Excessively high P/FCF: {pfcf:.2f}")
        else:
            details.append("FCF is zero or negative; P/FCF calculation skipped")
    else:
        # If not available from utility, try to get from cash_flows data
        cash_flows = fundamental_data.get("cash_flows", {})
        fcf_values = []
        
        # Get FCF values
        for fcf_key in ["Free Cash Flow", "FCF"]:
            if fcf_key in cash_flows:
                fcf_values = cash_flows[fcf_key]
                fcf_values = [float(fcf) if fcf is not None else None for fcf in fcf_values]
                break
        
        if fcf_values and len(fcf_values) > 0 and fcf_values[0] is not None:
            recent_fcf = fcf_values[0]
            
            if recent_fcf > 0 and market_cap is not None:
                pfcf = market_cap / recent_fcf
                
                if pfcf < 20:
                    score += 2
                    details.append(f"Reasonable P/FCF: {pfcf:.2f}")
                elif pfcf < 30:
                    score += 1
                    details.append(f"Somewhat high P/FCF: {pfcf:.2f}")
                else:
                    details.append(f"Excessively high P/FCF: {pfcf:.2f}")
            else:
                details.append("FCF is zero or negative; P/FCF calculation skipped")
        else:
            details.append("No valid FCF data for P/FCF calculation")
    
    # Normalize score to 0-10 scale
    max_raw_score = 4  # Maximum possible raw score (2+2)
    final_score = min(10, score * 10 / max_raw_score)
    
    # Determine rating
    if final_score >= 7.5:
        rating = "Attractive"
    elif final_score >= 5:
        rating = "Fairly Valued"
    elif final_score >= 2.5:
        rating = "Somewhat Expensive"
    else:
        rating = "Expensive"
    
    # Generate summary
    summary = f"Valuation: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Attractively priced relative to earnings and cash flow."
    elif final_score >= 5:
        summary += ". Reasonably priced given growth prospects."
    elif final_score >= 2.5:
        summary += ". Somewhat expensive but may be justified for a quality company."
    else:
        summary += ". Expensive valuation requiring exceptional growth to justify."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

class PhilFisherAgnoAgent:
    """
    Phil Fisher investment agent based on Agno framework.
    Implements Fisher's investment philosophy focusing on high-quality growth stocks.
    """
    
    def __init__(self, model_name: str = "sufe-aiflm-lab_fin-r1", model_provider: str = "lmstudio"):
        """Initialize the agent with the specified model."""
        self.model_name = model_name
        self.model_provider = model_provider
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent using the Agno framework."""
        self.agent = Agent(
            model=LMStudio(id=self.model_name, reasoning_effort="high"),
            markdown=True,
            reasoning=True,
        )
    
    def _prepare_agent_prompt(self, ticker: str, growth_analysis: Dict, margin_analysis: Dict, 
                              management_analysis: Dict, valuation_analysis: Dict) -> str:
        """Prepare the prompt for the agent with all analyses."""
        prompt = f"""
        You are Phil Fisher, one of the most influential investors of all time known for your growth investment approach. 
        You popularized "scuttlebutt" investment research and wrote the classic book "Common Stocks and Uncommon Profits".
        
        Please analyze the company {ticker} based on your investment philosophy. 
        Here are the key analyses to consider:
        
        GROWTH QUALITY ANALYSIS:
        {growth_analysis['summary']}
        Details: {growth_analysis['details']}
        
        MARGIN STABILITY ANALYSIS:
        {margin_analysis['summary']}
        Details: {margin_analysis['details']}
        
        MANAGEMENT EFFICIENCY ANALYSIS:
        {management_analysis['summary']}
        Details: {management_analysis['details']}
        
        VALUATION ANALYSIS:
        {valuation_analysis['summary']}
        Details: {valuation_analysis['details']}
        
        Based on this information and your investment philosophy, provide:
        1. A concise assessment of this company as an investment
        2. A clear rating: buy, hold, or sell
        3. Your rationale for this conclusion
        
        Your response should be in JSON format with keys: "assessment", "rating", and "rationale".
        """
        return prompt
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete analysis of stocks using Phil Fisher's principles.
        
        Args:
            state: Dictionary containing tickers and metadata
            
        Returns:
            Dictionary with analysis results and final signals
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "phil_fisher_agno_agent"

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
                print(f"Starting Phil Fisher analysis for {ticker}...")
                
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
                
                # Get market cap for various analyses
                market_cap_data = get_market_cap(ticker)
                if isinstance(market_cap_data, dict) and "current_market_cap" in market_cap_data:
                    market_cap = market_cap_data["current_market_cap"]
                elif isinstance(market_cap_data, (int, float)):
                    market_cap = market_cap_data
                else:
                    market_cap = None
                    print("Warning: Market cap data not available")
                
                # Perform all analyses
                print("Analyzing growth quality...")
                growth_analysis = analyze_growth_quality(ticker)
                
                print("Analyzing margin stability...")
                margin_analysis = analyze_margins_stability(ticker)
                
                print("Analyzing management efficiency...")
                management_analysis = analyze_management_efficiency(ticker)
                
                print("Analyzing valuation...")
                valuation_analysis = analyze_valuation(ticker, market_cap)
                
                # Calculate overall score (weighted average of all analyses)
                weights = {
                    "growth": 0.35,  # Fisher emphasized growth heavily
                    "margins": 0.25,  # Margins indicate business quality
                    "management": 0.25,  # Quality management is crucial
                    "valuation": 0.15,  # Fisher believed in paying fair price for quality
                }
                
                overall_score = (
                    growth_analysis["score"] * weights["growth"] +
                    margin_analysis["score"] * weights["margins"] +
                    management_analysis["score"] * weights["management"] +
                    valuation_analysis["score"] * weights["valuation"]
                )
                
                # Generate summary based on Fisher's philosophy
                if overall_score >= 7.5:
                    overall_rating = "Strong Buy"
                    signal = "bullish"
                elif overall_score >= 6:
                    overall_rating = "Buy"
                    signal = "bullish"
                elif overall_score >= 5:
                    overall_rating = "Hold"
                    signal = "neutral"
                elif overall_score >= 3.5:
                    overall_rating = "Sell"
                    signal = "bearish"
                else:
                    overall_rating = "Strong Sell"
                    signal = "bearish"
                
                analysis = {
                    "growth_quality": growth_analysis,
                    "margin_stability": margin_analysis,
                    "management_efficiency": management_analysis,
                    "valuation": valuation_analysis,
                    "overall_score": overall_score,
                    "overall_rating": overall_rating,
                    "signal": signal
                }
                
                # Debug print the analysis
                print(f"Analysis for {ticker}:")
                print(json.dumps(analysis, indent=2))
                
                # Prepare prompt and run LLM
                prompt = self._prepare_agent_prompt(
                    ticker, 
                    growth_analysis, 
                    margin_analysis,
                    management_analysis,
                    valuation_analysis
                )
                
                # Run the agent to get qualitative analysis
                llm_response = self.agent.run(prompt)
                parsed_response = self._parse_response(llm_response, ticker)
                
                # Store results
                results[ticker] = {
                    "signal": parsed_response.signal,
                    "confidence": parsed_response.confidence,
                    "reasoning": parsed_response.reasoning,
                    "analysis": analysis,
                    "data_availability": data_availability
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {"error": f"Error analyzing {ticker}: {str(e)}"}
        
        return {agent_name: results}
    
    def _parse_response(self, response: str, ticker: str) -> PhilFisherSignal:
        """Parse the response from the language model."""
        try:
            # Try to extract JSON if it exists
            import re
            import json
            
            # Look for JSON content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                
                # Convert rating to signal
                signal = "neutral"
                if "rating" in parsed_data:
                    rating = parsed_data["rating"].lower()
                    if "buy" in rating:
                        signal = "bullish"
                    elif "sell" in rating:
                        signal = "bearish"
                    
                return PhilFisherSignal(
                    signal=signal,
                    confidence=0.8,  # Default high confidence for structured response
                    reasoning=parsed_data.get("rationale", parsed_data.get("assessment", "No explanation provided"))
                )
            
            # If no JSON block with markers, try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                
                # Convert rating to signal
                signal = "neutral"
                if "rating" in parsed_data:
                    rating = parsed_data["rating"].lower()
                    if "buy" in rating:
                        signal = "bullish"
                    elif "sell" in rating:
                        signal = "bearish"
                    
                return PhilFisherSignal(
                    signal=signal,
                    confidence=0.8,  # Default high confidence for structured response
                    reasoning=parsed_data.get("rationale", parsed_data.get("assessment", "No explanation provided"))
                )
            
            # If all else fails, try to extract signal from text
            response_text = response.lower()
            if "buy" in response_text or "bullish" in response_text:
                signal = "bullish"
            elif "sell" in response_text or "bearish" in response_text:
                signal = "bearish"
            else:
                signal = "neutral"
            
            return PhilFisherSignal(
                signal=signal,
                confidence=0.5,  # Lower confidence for unstructured response
                reasoning=response[:1000] if len(response) > 1000 else response
            )
        except Exception as e:
            print(f"Error parsing response: {e}")
            return PhilFisherSignal(
                signal="neutral",
                confidence=0.0,
                reasoning=f"Error parsing response: {str(e)}"
            )
    
    def analyze(self, ticker: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a stock based on Phil Fisher's investment criteria.
        
        Args:
            ticker: Stock symbol to analyze
            end_date: Optional date to use for analysis (for historical analysis)
            
        Returns:
            Dictionary with analysis results and final signal
        """
        # Run the analysis
        state = {
            "data": {
                "tickers": [ticker],
                "end_date": end_date
            },
            "metadata": {}
        }
        result = self.run(state)
        
        return result


# Test function if run directly
if __name__ == "__main__":
    # Test example
    ticker = "HDFCBANK"  # Sample ticker
    agent = PhilFisherAgnoAgent(model_name="gpt-4o")  # Use a compatible model
    result = agent.analyze(ticker)
    
    # Print JSON result
    import json
    print(json.dumps(result, indent=2))