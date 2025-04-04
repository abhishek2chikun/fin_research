"""
Peter Lynch Investing Agent using Agno Framework
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
class PeterLynchSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str 

def analyze_growth(ticker: str) -> Dict[str, Any]:
    """
    Analyze growth according to Peter Lynch's categories:
    - Slow Growers: Large, mature companies growing slightly faster than GDP
    - Stalwarts: Large, dependable companies growing at 10-12% annually
    - Fast Growers: Small, aggressive companies growing at 20-25%+ annually
    - Cyclicals: Companies with sales and profits tied to economic cycles
    - Turnarounds: Companies recovering from financial distress
    - Asset Plays: Companies with valuable assets not reflected in stock price
    
    This function focuses on identifying growth rates and categorization.
    """
    score = 0
    details = []
    growth_category = "Unknown"
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for growth analysis",
            "rating": "Poor",
            "category": "Unknown",
            "summary": "Growth: Poor (0/10 points). Insufficient data."
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    balance_sheet = fundamental_data.get("balance_sheet", {})
    
    # 1. Analyze Revenue Growth (YoY)
    revenue_values = []
    if "Revenue" in profit_loss:
        revenue_values = profit_loss["Revenue"]
        revenue_values = [float(r) if r is not None else None for r in revenue_values]
        
    if revenue_values and len([r for r in revenue_values if r is not None]) >= 2:
        valid_revenues = [r for r in revenue_values if r is not None]
        
        # Calculate multi-period revenue growth
        latest_rev = valid_revenues[0]
        oldest_rev = valid_revenues[-1]
        
        if oldest_rev > 0:
            rev_growth = (latest_rev - oldest_rev) / oldest_rev
            
            # Align with original implementation thresholds
            if rev_growth > 0.25:  # 25%+ growth
                score += 3
                details.append(f"Strong revenue growth: {rev_growth:.1%}")
                growth_category = "Fast Grower"
            elif rev_growth > 0.10:  # 10-25% growth
                score += 2
                details.append(f"Moderate revenue growth: {rev_growth:.1%}")
                growth_category = "Stalwart"
            elif rev_growth > 0.02:  # 2-10% growth
                score += 1
                details.append(f"Slight revenue growth: {rev_growth:.1%}")
                growth_category = "Slow Grower"
            else:
                details.append(f"Flat or negative revenue growth: {rev_growth:.1%}")
                
            # Also calculate annual growth rates for consistency check
            growth_rates = []
            for i in range(1, len(valid_revenues)):
                if valid_revenues[i] > 0:
                    growth_rate = (valid_revenues[i-1] - valid_revenues[i]) / valid_revenues[i]
                    growth_rates.append(growth_rate)
                    
            if growth_rates:
                growth_consistency = statistics.pstdev(growth_rates) if len(growth_rates) > 1 else 0
                if growth_consistency < 0.05:  # Very consistent growth
                    score += 2
                    details.append(f"Exceptionally consistent growth (std dev: {growth_consistency:.2f})")
                elif growth_consistency < 0.10:  # Reasonably consistent
                    score += 1
                    details.append(f"Reasonably consistent growth (std dev: {growth_consistency:.2f})")
                else:
                    details.append(f"Inconsistent growth pattern (std dev: {growth_consistency:.2f})")
        else:
            details.append("Older revenue is zero/negative; can't compute revenue growth.")
    else:
        details.append("Not enough revenue data to assess growth.")
    
    # 2. Analyze EPS Growth (YoY)
    eps_values = []
    for eps_key in ["EPS", "Earnings Per Share", "Diluted EPS"]:
        if eps_key in profit_loss:
            eps_values = profit_loss[eps_key]
            eps_values = [float(e) if e is not None else None for e in eps_values]
            break
    
    if eps_values and len([e for e in eps_values if e is not None]) >= 2:
        valid_eps = [e for e in eps_values if e is not None]
        
        latest_eps = valid_eps[0]
        oldest_eps = valid_eps[-1]
        
        if abs(oldest_eps) > 1e-9:  # Ensure denominator isn't too close to zero
            eps_growth = (latest_eps - oldest_eps) / abs(oldest_eps)
            
            # Align with original implementation thresholds
            if eps_growth > 0.25:
                score += 3
                details.append(f"Strong EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.10:
                score += 2
                details.append(f"Moderate EPS growth: {eps_growth:.1%}")
            elif eps_growth > 0.02:
                score += 1
                details.append(f"Slight EPS growth: {eps_growth:.1%}")
            else:
                details.append(f"Minimal or negative EPS growth: {eps_growth:.1%}")
        else:
            details.append("Older EPS is near zero; skipping EPS growth calculation.")
    else:
        details.append("Not enough EPS data for growth calculation.")
    
    # 3. Check for signs of a potential turnaround
    net_income_values = []
    for ni_key in ["Net Income", "Net Profit", "Profit after Tax"]:
        if ni_key in profit_loss:
            net_income_values = profit_loss[ni_key]
            net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
            break
    
    if net_income_values and len([ni for ni in net_income_values if ni is not None]) >= 2:
        valid_ni = [ni for ni in net_income_values if ni is not None]
        
        # Check for recent improvement from negative to positive earnings
        if len(valid_ni) >= 3 and valid_ni[0] > 0 and valid_ni[1] < 0:
            growth_category = "Turnaround"
            score += 3
            details.append("Turnaround: Recent transition from negative to positive earnings")
        elif len(valid_ni) >= 3 and valid_ni[0] > valid_ni[1] and valid_ni[1] < 0 and valid_ni[2] < 0:
            growth_category = "Turnaround"
            score += 2
            details.append("Potential Turnaround: Improving losses, not yet profitable")
    
    # 4. Check for signs of cyclicality
    if revenue_values and len(revenue_values) >= 5:
        valid_rev = [r for r in revenue_values if r is not None]
        
        # Simple heuristic: alternating ups and downs may indicate cyclicality
        ups_and_downs = 0
        for i in range(1, len(valid_rev) - 1):
            if (valid_rev[i] > valid_rev[i-1] and valid_rev[i] > valid_rev[i+1]) or (valid_rev[i] < valid_rev[i-1] and valid_rev[i] < valid_rev[i+1]):
                ups_and_downs += 1
                
        if ups_and_downs >= 2 and growth_category == "Unknown":
            growth_category = "Cyclical"
            score += 1
            details.append("Cyclical: Revenue shows cyclical pattern with multiple peaks and troughs")
    
    # 5. Check for Asset Play potential
    if balance_sheet:
        # Lynch looked for hidden assets like real estate, intellectual property, etc.
        # This is a simplified check for high book value relative to market cap
        total_assets = None
        for asset_key in ["Total Assets", "Assets"]:
            if asset_key in balance_sheet and balance_sheet[asset_key]:
                total_assets = balance_sheet[asset_key][0]
                if total_assets is not None:
                    total_assets = float(total_assets)
                break
                
        if total_assets:
            market_cap_data = get_market_cap(ticker)
            market_cap = None
            if isinstance(market_cap_data, dict) and "current_market_cap" in market_cap_data:
                market_cap = market_cap_data["current_market_cap"]
            elif isinstance(market_cap_data, (int, float)):
                market_cap = market_cap_data
                
            if market_cap and market_cap > 0:
                assets_to_market_ratio = total_assets / market_cap
                
                if assets_to_market_ratio > 1.5 and growth_category == "Unknown":
                    growth_category = "Asset Play"
                    score += 2
                    details.append(f"Asset Play: Assets-to-market ratio of {assets_to_market_ratio:.2f}")
                elif assets_to_market_ratio > 1.0 and growth_category == "Unknown":
                    growth_category = "Asset Play"
                    score += 1
                    details.append(f"Potential Asset Play: Assets-to-market ratio of {assets_to_market_ratio:.2f}")
    
    # Normalize score to 0-10 scale - use a max of 6 points like the original implementation
    max_raw_score = 6
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
    
    # If no category was determined, provide a default
    if growth_category == "Unknown":
        if final_score >= 5:
            growth_category = "Stalwart"
        else:
            growth_category = "Slow Grower"
    
    # Generate summary
    summary = f"Growth: {rating} ({final_score:.1f}/10 points)"
    summary += f". Category: {growth_category}."
    
    if growth_category == "Fast Grower":
        summary += " High growth company with significant expansion potential."
    elif growth_category == "Stalwart":
        summary += " Steady, reliable growth with moderate upside."
    elif growth_category == "Slow Grower":
        summary += " Mature company with modest growth prospects."
    elif growth_category == "Turnaround":
        summary += " Company showing potential recovery from difficulties."
    elif growth_category == "Cyclical":
        summary += " Performance tied to economic or industry cycles."
    elif growth_category == "Asset Play":
        summary += " Valuable assets potentially underappreciated by the market."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "category": growth_category,
        "summary": summary
    } 

def analyze_valuation(ticker: str, growth_category: str, market_cap: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze valuation according to Peter Lynch's principles:
    - PEG Ratio (Price/Earnings to Growth ratio)
    - Appropriate P/E based on growth category
    - Reasonable debt levels
    
    Lynch famously believed a fair P/E ratio should roughly equal the company's growth rate,
    making a PEG ratio of 1.0 the sweet spot.
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
    balance_sheet = fundamental_data.get("balance_sheet", {})
    
    # 1. Get P/E ratio for calculation
    pe_ratio = None
    
    # Get P/E from utility function
    pe_data = get_pe_ratio(ticker)
    if pe_data and "current_pe" in pe_data and pe_data["current_pe"] is not None:
        pe_ratio = pe_data["current_pe"]
        details.append(f"Estimated P/E: {pe_ratio:.2f}")
    else:
        # Calculate manually if needed using Net Income
        net_income_values = []
        for income_key in ["Net Income", "Net Profit", "Profit After Tax"]:
            if income_key in profit_loss:
                net_income_values = profit_loss[income_key]
                net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
                break
                
        if net_income_values and len(net_income_values) > 0 and net_income_values[0] is not None:
            recent_net_income = net_income_values[0]
            if recent_net_income > 0 and market_cap is not None:
                pe_ratio = market_cap / recent_net_income
                details.append(f"Estimated P/E: {pe_ratio:.2f}")
            else:
                details.append("No positive net income => can't compute approximate P/E")
        else:
            details.append("Net income data not available to calculate P/E")
    
    # 2. Calculate growth rate for PEG calculation
    growth_rate = None
    # First try EPS growth
    eps_values = []
    for eps_key in ["EPS", "Earnings Per Share", "Diluted EPS"]:
        if eps_key in profit_loss:
            eps_values = profit_loss[eps_key]
            eps_values = [float(e) if e is not None else None for e in eps_values]
            break
    
    if eps_values and len([e for e in eps_values if e is not None]) >= 2:
        valid_eps = [e for e in eps_values if e is not None]
        latest_eps = valid_eps[0]
        oldest_eps = valid_eps[-1]
        if oldest_eps > 0:
            eps_growth_rate = (latest_eps - oldest_eps) / oldest_eps
            growth_rate = eps_growth_rate
            details.append(f"Approx EPS growth rate: {eps_growth_rate:.1%}")
        else:
            details.append("Cannot compute EPS growth rate (older EPS <= 0)")
    else:
        details.append("Not enough EPS data to compute growth rate")
    
    # If EPS growth not available, try revenue growth
    if growth_rate is None:
        revenue_values = []
        if "Revenue" in profit_loss:
            revenue_values = profit_loss["Revenue"]
            revenue_values = [float(r) if r is not None else None for r in revenue_values]
            
        if revenue_values and len([r for r in revenue_values if r is not None]) >= 2:
            valid_revenues = [r for r in revenue_values if r is not None]
            latest_rev = valid_revenues[0]
            oldest_rev = valid_revenues[-1]
            if oldest_rev > 0:
                rev_growth_rate = (latest_rev - oldest_rev) / oldest_rev
                growth_rate = rev_growth_rate
                details.append(f"Using revenue growth rate: {rev_growth_rate:.1%}")
    
    # Also try to get from utility function
    if growth_rate is None:
        growth_data = get_revenue_growth(ticker)
        if growth_data and "growth_rate" in growth_data and growth_data["growth_rate"] is not None:
            growth_rate = growth_data["growth_rate"]
            details.append(f"Using utility-provided growth rate: {growth_rate:.1%}")
    
    # 3. Calculate PEG if possible
    peg_ratio = None
    if pe_ratio is not None and growth_rate is not None and growth_rate > 0:
        # Convert growth rate to percentage for PEG calculation
        peg_ratio = pe_ratio / (growth_rate * 100)
        details.append(f"PEG ratio: {peg_ratio:.2f}")
        
        # Score based on PEG ratio (Lynch's key metric)
        if peg_ratio < 1.0:
            score += 3
            details.append("Excellent PEG ratio (below 1.0)")
        elif peg_ratio < 2.0:
            score += 2
            details.append("Good PEG ratio (between 1.0 and 2.0)")
        elif peg_ratio < 3.0:
            score += 1
            details.append("Fair PEG ratio (between 2.0 and 3.0)")
        else:
            details.append("Poor PEG ratio (above 3.0)")
    else:
        details.append("Unable to calculate PEG ratio (missing P/E or growth rate)")
    
    # 4. Assess P/E ratio independently based on Lynch's classifications
    if pe_ratio is not None:
        if pe_ratio < 15:
            score += 2
            details.append("Low P/E ratio (below 15)")
        elif pe_ratio < 25:
            score += 1
            details.append("Moderate P/E ratio (15-25)")
        else:
            details.append("High P/E ratio (above 25)")
    
    # 5. Evaluate debt levels (Lynch preferred companies with low debt)
    debt_equity_data = get_debt_to_equity(ticker)
    if debt_equity_data and "current_ratio" in debt_equity_data and debt_equity_data["current_ratio"] is not None:
        debt_equity = debt_equity_data["current_ratio"]
        
        if debt_equity < 0.3:
            score += 3
            details.append(f"Very low debt-to-equity ratio: {debt_equity:.2f}")
        elif debt_equity < 1.0:
            score += 2
            details.append(f"Reasonable debt-to-equity ratio: {debt_equity:.2f}")
        elif debt_equity < 2.0:
            score += 1
            details.append(f"Moderate debt-to-equity ratio: {debt_equity:.2f}")
        else:
            details.append(f"High debt-to-equity ratio: {debt_equity:.2f}")
    else:
        # Try to calculate manually
        total_debt = None
        total_equity = None
        
        for debt_key in ["Total Debt", "Long Term Debt", "Total Long Term Debt"]:
            if debt_key in balance_sheet and balance_sheet[debt_key]:
                total_debt = balance_sheet[debt_key][0]
                if total_debt is not None:
                    total_debt = float(total_debt)
                break
                
        for equity_key in ["Total Equity", "Shareholders' Equity", "Total Shareholders' Equity"]:
            if equity_key in balance_sheet and balance_sheet[equity_key]:
                total_equity = balance_sheet[equity_key][0]
                if total_equity is not None:
                    total_equity = float(total_equity)
                break
                
        if total_debt is not None and total_equity is not None and total_equity > 0:
            debt_equity = total_debt / total_equity
            
            if debt_equity < 0.3:
                score += 3
                details.append(f"Very low debt-to-equity ratio: {debt_equity:.2f}")
            elif debt_equity < 1.0:
                score += 2
                details.append(f"Reasonable debt-to-equity ratio: {debt_equity:.2f}")
            elif debt_equity < 2.0:
                score += 1
                details.append(f"Moderate debt-to-equity ratio: {debt_equity:.2f}")
            else:
                details.append(f"High debt-to-equity ratio: {debt_equity:.2f}")
        else:
            details.append("Debt-to-equity ratio not available")
    
    # Normalize score to 0-10 scale - use max_raw_score of 5 like the original implementation
    max_raw_score = 5
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
        summary += ". Attractively priced with strong growth-to-value relationship."
    elif final_score >= 5:
        summary += ". Reasonably priced relative to growth and category."
    elif final_score >= 2.5:
        summary += ". Price somewhat high relative to growth potential."
    else:
        summary += ". Expensive valuation requiring exceptional performance to justify."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

def analyze_business_quality(ticker: str) -> Dict[str, Any]:
    """
    Analyze business quality according to Peter Lynch's principles:
    - Low debt-to-equity
    - Strong operating margin or gross margin
    - Positive free cash flow
    - Ease of understanding the business ("if a six-year-old can't understand it, don't invest")
    
    Lynch preferred simple businesses with strong fundamentals and understandable advantages.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for business quality analysis",
            "rating": "Poor",
            "summary": "Business Quality: Poor (0/10 points). Insufficient data."
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    balance_sheet = fundamental_data.get("balance_sheet", {})
    cash_flows = fundamental_data.get("cash_flows", {})
    
    # 1. Analyze Debt-to-Equity (Lynch avoided heavily indebted businesses)
    debt_equity_data = get_debt_to_equity(ticker)
    if debt_equity_data and "current_ratio" in debt_equity_data and debt_equity_data["current_ratio"] is not None:
        debt_equity = debt_equity_data["current_ratio"]
        
        if debt_equity < 0.5:
            score += 2
            details.append(f"Low debt-to-equity: {debt_equity:.2f}")
        elif debt_equity < 1.0:
            score += 1
            details.append(f"Moderate debt-to-equity: {debt_equity:.2f}")
        else:
            details.append(f"High debt-to-equity: {debt_equity:.2f}")
    else:
        # Try to calculate manually
        total_debt = None
        total_equity = None
        
        for debt_key in ["Total Debt", "Long Term Debt", "Total Long Term Debt"]:
            if debt_key in balance_sheet and balance_sheet[debt_key]:
                total_debt = balance_sheet[debt_key][0]
                if total_debt is not None:
                    total_debt = float(total_debt)
                break
                
        for equity_key in ["Total Equity", "Shareholders' Equity", "Total Shareholders' Equity"]:
            if equity_key in balance_sheet and balance_sheet[equity_key]:
                total_equity = balance_sheet[equity_key][0]
                if total_equity is not None:
                    total_equity = float(total_equity)
                break
                
        if total_debt is not None and total_equity is not None and total_equity > 0:
            debt_equity = total_debt / total_equity
            
            if debt_equity < 0.5:
                score += 2
                details.append(f"Low debt-to-equity: {debt_equity:.2f}")
            elif debt_equity < 1.0:
                score += 1
                details.append(f"Moderate debt-to-equity: {debt_equity:.2f}")
            else:
                details.append(f"High debt-to-equity: {debt_equity:.2f}")
        else:
            details.append("No consistent debt/equity data available")
    
    # 2. Analyze Operating Margin (Lynch liked businesses with pricing power)
    margin_data = get_operating_margin(ticker)
    if margin_data and "current_margin" in margin_data and margin_data["current_margin"] is not None:
        operating_margin = margin_data["current_margin"]
        
        if operating_margin > 0.20:
            score += 2
            details.append(f"Strong operating margin: {operating_margin:.1%}")
        elif operating_margin > 0.10:
            score += 1
            details.append(f"Moderate operating margin: {operating_margin:.1%}")
        else:
            details.append(f"Low operating margin: {operating_margin:.1%}")
    else:
        # Try to calculate from financial data
        revenue_values = []
        operating_income_values = []
        
        if "Revenue" in profit_loss and profit_loss["Revenue"]:
            revenue_values = profit_loss["Revenue"]
            revenue_values = [float(r) if r is not None else None for r in revenue_values]
        
        for oi_key in ["Operating Income", "EBIT", "Operating Profit"]:
            if oi_key in profit_loss and profit_loss[oi_key]:
                operating_income_values = profit_loss[oi_key]
                operating_income_values = [float(oi) if oi is not None else None for oi in operating_income_values]
                break
        
        if (revenue_values and operating_income_values and 
            len(revenue_values) > 0 and len(operating_income_values) > 0 and
            revenue_values[0] is not None and operating_income_values[0] is not None and
            revenue_values[0] > 0):
            
            operating_margin = operating_income_values[0] / revenue_values[0]
            
            if operating_margin > 0.20:
                score += 2
                details.append(f"Strong operating margin: {operating_margin:.1%}")
            elif operating_margin > 0.10:
                score += 1
                details.append(f"Moderate operating margin: {operating_margin:.1%}")
            else:
                details.append(f"Low operating margin: {operating_margin:.1%}")
        else:
            details.append("No operating margin data available")
    
    # 3. Analyze Free Cash Flow (Lynch valued positive FCF)
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "current_fcf" in fcf_data and fcf_data["current_fcf"] is not None:
        recent_fcf = fcf_data["current_fcf"]
        
        if recent_fcf > 0:
            score += 2
            details.append(f"Positive free cash flow: {recent_fcf:,.0f}")
        else:
            details.append(f"Negative free cash flow: {recent_fcf:,.0f}")
    else:
        # Try to get from cash flow statement
        fcf_values = []
        for fcf_key in ["Free Cash Flow", "FCF"]:
            if fcf_key in cash_flows and cash_flows[fcf_key]:
                fcf_values = cash_flows[fcf_key]
                fcf_values = [float(fcf) if fcf is not None else None for fcf in fcf_values]
                break
        
        if fcf_values and len(fcf_values) > 0 and fcf_values[0] is not None:
            recent_fcf = fcf_values[0]
            
            if recent_fcf > 0:
                score += 2
                details.append(f"Positive free cash flow: {recent_fcf:,.0f}")
            else:
                details.append(f"Negative free cash flow: {recent_fcf:,.0f}")
        else:
            details.append("No free cash flow data available")
    
    # 4. Analyze Cash Position (bonus, not in original implementation)
    cash_values = []
    for cash_key in ["Cash and Cash Equivalents", "Cash & Cash Equivalents", "Cash"]:
        if cash_key in balance_sheet and balance_sheet[cash_key]:
            cash_values = balance_sheet[cash_key]
            cash_values = [float(c) if c is not None else None for c in cash_values]
            break
    
    if cash_values and len(cash_values) > 0 and cash_values[0] is not None:
        current_cash = cash_values[0]
        
        # Compare cash to total assets to gauge liquidity
        total_assets = None
        for asset_key in ["Total Assets", "Assets"]:
            if asset_key in balance_sheet and balance_sheet[asset_key]:
                total_assets_values = balance_sheet[asset_key]
                if total_assets_values and len(total_assets_values) > 0 and total_assets_values[0] is not None:
                    total_assets = float(total_assets_values[0])
                break
        
        if total_assets and total_assets > 0:
            cash_to_assets = current_cash / total_assets
            
            if cash_to_assets >= 0.2:  # 20%+ of assets in cash
                score += 1
                details.append(f"Strong cash position: {cash_to_assets:.1%} of assets")
            elif cash_to_assets >= 0.1:  # 10%+ of assets in cash
                score += 0.5
                details.append(f"Adequate cash position: {cash_to_assets:.1%} of assets")
            else:
                details.append(f"Limited cash position: {cash_to_assets:.1%} of assets")
    
    # Normalize score to 0-10 scale using a max_raw_score of 6 like original implementation
    max_raw_score = 6
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
    summary = f"Business Quality: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Excellent fundamentals with strong margins and low debt."
    elif final_score >= 5:
        summary += ". Good fundamentals with reasonable margins and manageable debt."
    elif final_score >= 2.5:
        summary += ". Average fundamentals with some concerns about margins or debt."
    else:
        summary += ". Poor fundamentals with significant concerns about financial health."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    } 

class PeterLynchAgnoAgent:
    """
    Peter Lynch investment agent based on Agno framework.
    Implements Lynch's investment philosophy focused on understanding businesses,
    categorizing companies by growth types, and using the PEG ratio for valuation.
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
    
    def _prepare_agent_prompt(self, ticker: str, growth_analysis: Dict, business_quality: Dict, 
                              valuation_analysis: Dict) -> str:
        """Prepare the prompt for the agent with all analyses."""
        prompt = f"""
        You are Peter Lynch, one of the most successful investors of all time. 
        You managed the Magellan Fund at Fidelity and achieved a 29.2% annual return between 1977 and 1990.
        Your investment philosophy is captured in the books "One Up on Wall Street" and "Beating the Street".
        
        Your key principles include:
        1. Invest in what you know and understand
        2. Look for businesses a six-year-old could understand
        3. Categorize companies (Slow Growers, Stalwarts, Fast Growers, Cyclicals, Turnarounds, Asset Plays)
        4. Focus on PEG ratio (P/E ratio divided by growth rate)
        5. Invest for the long term, but regularly review holdings
        
        Please analyze the company {ticker} based on your investment philosophy. 
        Here are the key analyses to consider:
        
        GROWTH CLASSIFICATION:
        {growth_analysis['summary']}
        Details: {growth_analysis['details']}
        Category: {growth_analysis['category']}
        
        BUSINESS QUALITY ANALYSIS:
        {business_quality['summary']}
        Details: {business_quality['details']}
        
        VALUATION ANALYSIS:
        {valuation_analysis['summary']}
        Details: {valuation_analysis['details']}
        
        Based on this information and your investment philosophy, provide:
        1. A concise assessment of this company as an investment
        2. A clear rating: buy, hold, or sell
        3. Your confidence level (0.0 to 1.0)
        4. Your rationale for this conclusion, tying back to your investment principles
        
        Your response should be in JSON format with keys: "assessment", "rating", "confidence", and "rationale".
        """
        return prompt
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete analysis of stocks using Peter Lynch's principles.
        
        Args:
            state: Dictionary containing tickers and metadata
            
        Returns:
            Dictionary with analysis results and final signals
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "peter_lynch_agno_agent"

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
                print(f"Starting Peter Lynch analysis for {ticker}...")
                
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
                print("Analyzing growth classification...")
                growth_analysis = analyze_growth(ticker)
                
                print("Analyzing business quality...")
                business_quality = analyze_business_quality(ticker)
                
                growth_category = growth_analysis.get("category", "Unknown")
                print(f"Growth category determined: {growth_category}")
                
                print("Analyzing valuation...")
                valuation_analysis = analyze_valuation(ticker, growth_category, market_cap)
                
                # Combine analyses into a single object
                analysis = {
                    "growth": growth_analysis,
                    "business_quality": business_quality,
                    "valuation": valuation_analysis
                }
                
                # Debug print the analysis
                print(f"Analysis for {ticker}:")
                print(json.dumps(analysis, indent=2))
                
                # Prepare prompt and run LLM to get Peter Lynch-style assessment
                prompt = self._prepare_agent_prompt(
                    ticker, 
                    growth_analysis, 
                    business_quality,
                    valuation_analysis
                )
                
                # Run the LLM model to get qualitative assessment
                print("Running LLM analysis...")
                llm_response = self.agent.run(prompt)
                
                # Parse LLM response into structured signal
                parsed_response = self._parse_response(llm_response, ticker)
                
                # Store results including both analysis data and LLM signal
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
    
    def _parse_response(self, response: str, ticker: str) -> PeterLynchSignal:
        """Parse the response from the language model."""
        try:
            # Check if response is a string or an Agno response object
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'message'):
                response_text = response.message.content
            else:
                # Try to convert to string if it's not already
                response_text = str(response)
            
            # Try to extract JSON if it exists
            import re
            import json
            
            # Look for JSON content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
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
                
                # Get confidence directly or default
                confidence = 0.8  # Default
                if "confidence" in parsed_data and isinstance(parsed_data["confidence"], (int, float)):
                    confidence = float(parsed_data["confidence"])
                    
                return PeterLynchSignal(
                    signal=signal,
                    confidence=confidence,
                    reasoning=parsed_data.get("rationale", parsed_data.get("assessment", "No explanation provided"))
                )
            
            # If no JSON block with markers, try to find raw JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
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
                    
                    # Get confidence directly or default
                    confidence = 0.8  # Default
                    if "confidence" in parsed_data and isinstance(parsed_data["confidence"], (int, float)):
                        confidence = float(parsed_data["confidence"])
                        
                    return PeterLynchSignal(
                        signal=signal,
                        confidence=confidence,
                        reasoning=parsed_data.get("rationale", parsed_data.get("assessment", "No explanation provided"))
                    )
                except json.JSONDecodeError:
                    # If JSON parsing fails, continue to text-based analysis
                    pass
            
            # If all else fails, try to extract signal from text
            if "buy" in response_text.lower() or "bullish" in response_text.lower():
                signal = "bullish"
            elif "sell" in response_text.lower() or "bearish" in response_text.lower():
                signal = "bearish"
            else:
                signal = "neutral"
            
            # Try to extract a confidence value
            confidence_matches = re.findall(r"confidence[:\s]+(\d+\.?\d*)", response_text.lower())
            confidence = float(confidence_matches[0]) if confidence_matches else 0.7
            
            # Make sure confidence is between 0 and 1
            if confidence > 1:
                confidence = confidence / 100.0 if confidence > 10 else 0.7
            
            return PeterLynchSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),
                reasoning=response_text[:1000] if len(response_text) > 1000 else response_text
            )
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Create a defensive signal based on growth and valuation scores
            return PeterLynchSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Based on quantitative metrics (without LLM analysis due to error: {str(e)})"
            )
    
    def analyze(self, ticker: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a stock based on Peter Lynch's investment criteria.
        
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
    agent = PeterLynchAgnoAgent()  # Use default LMStudio model
    result = agent.analyze(ticker)
    
    # Print JSON result
    import json
    print(json.dumps(result, indent=2))