"""
Warren Buffett Investing Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
import math
import re
import statistics
import numpy as np
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
class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def analyze_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Analyze company fundamentals based on Buffett's criteria:
    - Return on Equity (ROE)
    - Debt to Equity Ratio
    - Operating Margin
    - Current Ratio and other liquidity metrics
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient fundamental data",
            "rating": "Poor",
            "summary": "Fundamental Analysis: Poor (0/10 points). Insufficient data."
        }
    
    # Check ROE (Return on Equity)
    roe_data = get_roe(ticker)
    if roe_data and "current_roe" in roe_data and roe_data["current_roe"] is not None:
        current_roe = roe_data["current_roe"]
        if current_roe > 0.15:  # 15% ROE threshold - Buffett likes high ROE
            score += 2
            details.append(f"Strong ROE of {current_roe:.1%}")
        else:
            details.append(f"Weak ROE of {current_roe:.1%}")
    else:
        details.append("ROE data not available")
    
    # Check Debt to Equity
    debt_equity_data = get_debt_to_equity(ticker)
    if debt_equity_data and "current_ratio" in debt_equity_data and debt_equity_data["current_ratio"] is not None:
        debt_to_equity = debt_equity_data["current_ratio"]
        if debt_to_equity < 0.5:  # Buffett prefers companies with low debt
            score += 2
            details.append(f"Conservative debt levels (D/E ratio: {debt_to_equity:.2f})")
        elif debt_to_equity < 1.0:
            score += 1
            details.append(f"Moderate debt levels (D/E ratio: {debt_to_equity:.2f})")
        else:
            details.append(f"High debt to equity ratio of {debt_to_equity:.2f}")
    else:
        details.append("Debt to equity data not available")
    
    # Check Operating Margin
    margin_data = get_operating_margin(ticker)
    if margin_data and "current_operating_margin" in margin_data and margin_data["current_operating_margin"] is not None:
        operating_margin = margin_data["current_operating_margin"]
        if operating_margin > 0.15:  # Buffett likes high-margin businesses
            score += 2
            details.append(f"Strong operating margins of {operating_margin:.1%}")
        elif operating_margin > 0.10:
            score += 1
            details.append(f"Decent operating margins of {operating_margin:.1%}")
        else:
            details.append(f"Weak operating margin of {operating_margin:.1%}")
    else:
        details.append("Operating margin data not available")
    
    # Check Liquidity (Current Ratio or similar)
    balance_sheet = fundamental_data.get("balance_sheet", {})
    current_assets = None
    current_liabilities = None
    
    # Try to find current assets
    for key in ["Current Assets", "Total Current Assets"]:
        if key in balance_sheet:
            current_assets_values = balance_sheet[key]
            if current_assets_values and len(current_assets_values) > 0:
                current_assets = float(current_assets_values[0]) if current_assets_values[0] is not None else None
                break
    
    # Try to find current liabilities
    for key in ["Current Liabilities", "Total Current Liabilities"]:
        if key in balance_sheet:
            current_liabilities_values = balance_sheet[key]
            if current_liabilities_values and len(current_liabilities_values) > 0:
                current_liabilities = float(current_liabilities_values[0]) if current_liabilities_values[0] is not None else None
                break
    
    # Calculate current ratio if both metrics are available
    if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio > 1.5:  # Good liquidity threshold
            score += 1
            details.append(f"Good liquidity position (Current ratio: {current_ratio:.2f})")
        elif current_ratio > 1.0:
            score += 0.5
            details.append(f"Adequate liquidity (Current ratio: {current_ratio:.2f})")
        else:
            details.append(f"Weak liquidity with current ratio of {current_ratio:.2f}")
    else:
        details.append("Current ratio data not available")
    
    # Check Free Cash Flow (Buffett cares about cash generation)
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "current_fcf" in fcf_data and fcf_data["current_fcf"] is not None:
        current_fcf = fcf_data["current_fcf"]
        if current_fcf > 0:
            score += 2
            details.append(f"Positive free cash flow of {current_fcf:.2f}")
        else:
            details.append(f"Negative free cash flow of {current_fcf:.2f}")
    else:
        details.append("Free cash flow data not available")
    
    # Adjust score to 0-10 scale
    max_raw_score = 9  # Maximum possible score from criteria above
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
    summary = f"Fundamental Analysis: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Strong financial position with excellent business fundamentals."
    elif final_score >= 5:
        summary += ". Good business fundamentals with some financial strengths."
    elif final_score >= 2.5:
        summary += ". Average fundamentals with some areas of concern."
    else:
        summary += ". Weak fundamentals with significant financial concerns."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    }

def analyze_consistency(ticker: str) -> Dict[str, Any]:
    """
    Analyze earnings consistency and growth.
    Buffett values predictability and consistent performance over time.
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for consistency analysis",
            "rating": "Poor",
            "summary": "Consistency Analysis: Poor (0/10 points). Insufficient data."
        }
    
    # Check earnings growth trend
    profit_loss = fundamental_data.get("profit_loss", {})
    earnings_values = []
    
    # Look for net income/profit in profit_loss data
    for earnings_key in ["Net Income", "Net Profit", "Profit After Tax"]:
        if earnings_key in profit_loss:
            earnings_values = profit_loss[earnings_key]
            earnings_values = [float(e) if e is not None else None for e in earnings_values]
            break
    
    # Analyze earnings consistency if we have enough data
    if earnings_values and len([e for e in earnings_values if e is not None]) >= 4:
        # Remove None values
        valid_earnings = [e for e in earnings_values if e is not None]
        
        # Check for consistent growth pattern
        growth_periods = 0
        for i in range(len(valid_earnings) - 1):
            if valid_earnings[i] > valid_earnings[i + 1]:  # Note: index 0 is latest period
                growth_periods += 1
        
        growth_ratio = growth_periods / (len(valid_earnings) - 1)
        
        if growth_ratio >= 0.8:  # Consistent growth in 80%+ of periods
            score += 3
            details.append(f"Consistent earnings growth over {growth_periods}/{len(valid_earnings)-1} periods")
        elif growth_ratio >= 0.5:  # Growth in majority of periods
            score += 2
            details.append(f"Mostly consistent earnings growth ({growth_periods}/{len(valid_earnings)-1} periods)")
        else:
            details.append(f"Inconsistent earnings growth ({growth_periods}/{len(valid_earnings)-1} periods)")
        
        # Calculate total growth rate from oldest to latest (if both values are positive)
        if len(valid_earnings) >= 2 and valid_earnings[-1] > 0 and valid_earnings[0] > 0:
            growth_rate = (valid_earnings[0] - valid_earnings[-1]) / abs(valid_earnings[-1])
            if growth_rate > 0.10:  # More than 10% average annual growth
                score += 2
                details.append(f"Strong earnings growth of {growth_rate:.1%} over past {len(valid_earnings)} periods")
            elif growth_rate > 0:
                score += 1
                details.append(f"Positive earnings growth of {growth_rate:.1%} over past {len(valid_earnings)} periods")
            else:
                details.append(f"Earnings declined by {-growth_rate:.1%} over past {len(valid_earnings)} periods")
    else:
        details.append("Insufficient earnings data for trend analysis")
    
    # Check revenue consistency
    revenue_values = []
    if "Revenue" in profit_loss:
        revenue_values = profit_loss["Revenue"]
        revenue_values = [float(r) if r is not None else None for r in revenue_values]
    
    if revenue_values and len([r for r in revenue_values if r is not None]) >= 4:
        # Remove None values
        valid_revenue = [r for r in revenue_values if r is not None]
        
        # Check for consistent growth pattern
        growth_periods = 0
        for i in range(len(valid_revenue) - 1):
            if valid_revenue[i] > valid_revenue[i + 1]:  # Note: index 0 is latest period
                growth_periods += 1
        
        growth_ratio = growth_periods / (len(valid_revenue) - 1)
        
        if growth_ratio >= 0.8:  # Consistent growth in 80%+ of periods
            score += 2
            details.append(f"Consistent revenue growth over {growth_periods}/{len(valid_revenue)-1} periods")
        elif growth_ratio >= 0.5:  # Growth in majority of periods
            score += 1
            details.append(f"Mostly consistent revenue growth ({growth_periods}/{len(valid_revenue)-1} periods)")
        else:
            details.append(f"Inconsistent revenue growth ({growth_periods}/{len(valid_revenue)-1} periods)")
    else:
        details.append("Insufficient revenue data for trend analysis")
    
    # Check dividend consistency if relevant
    cash_flows = fundamental_data.get("cash_flows", {})
    dividend_values = []
    
    for dividend_key in ["Dividend Paid", "Dividends Paid", "Common Stock Dividend Paid"]:
        if dividend_key in cash_flows:
            dividend_values = cash_flows[dividend_key]
            dividend_values = [float(d) if d is not None else None for d in dividend_values]
            break
    
    if dividend_values and len([d for d in dividend_values if d is not None and d != 0]) >= 3:
        # Check if dividends are consistently paid
        valid_dividends = [d for d in dividend_values if d is not None]
        
        # Count number of periods with positive dividends (assuming dividend paid is shown as negative in cash flow)
        dividend_periods = sum(1 for d in valid_dividends if d < 0)
        
        if dividend_periods == len(valid_dividends):
            score += 1
            details.append("Consistent dividend payments across all periods")
        elif dividend_periods >= len(valid_dividends) * 0.75:
            score += 0.5
            details.append(f"Mostly consistent dividends ({dividend_periods}/{len(valid_dividends)} periods)")
        else:
            details.append(f"Inconsistent dividend payments ({dividend_periods}/{len(valid_dividends)} periods)")
    
    # Adjust score to 0-10 scale
    max_raw_score = 8  # Maximum possible score from criteria above
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
    summary = f"Consistency Analysis: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Highly consistent business performance with reliable growth."
    elif final_score >= 5:
        summary += ". Generally consistent business performance with some growth."
    elif final_score >= 2.5:
        summary += ". Mixed consistency with periods of growth and decline."
    else:
        summary += ". Inconsistent business performance with unreliable growth."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary
    }

def analyze_moat(ticker: str) -> Dict[str, Any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    Buffett places enormous emphasis on companies with wide moats.
    Key indicators:
    - Stability and high level of ROE over time
    - Stability and high level of operating margins over time
    - Pricing power
    - Brand strength
    - Network effects or switching costs
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for moat analysis",
            "rating": "Poor",
            "summary": "Moat Analysis: Poor (0/10 points). Insufficient data.",
            "max_score": 10
        }
    
    # 1. Check for stable and high ROE over time
    roe_data = get_roe(ticker)
    if roe_data and "historical_data" in roe_data and roe_data["historical_data"]:
        historical_roes = roe_data["historical_data"]
        
        if len(historical_roes) >= 3:
            # Count number of periods with ROE > 15%
            high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
            
            if high_roe_periods == len(historical_roes):
                score += 3
                details.append(f"Consistently high ROE >15% across all {len(historical_roes)} periods")
            elif high_roe_periods >= len(historical_roes) * 0.7:
                score += 2
                details.append(f"Strong ROE >15% in {high_roe_periods}/{len(historical_roes)} periods")
            elif high_roe_periods > 0:
                score += 1
                details.append(f"Occasional high ROE >15% in {high_roe_periods}/{len(historical_roes)} periods")
            else:
                details.append("No periods with ROE >15%")
            
            # Also check ROE stability (variation)
            if len(historical_roes) >= 4:
                try:
                    roe_std = statistics.stdev(historical_roes)
                    roe_mean = statistics.mean(historical_roes)
                    roe_cv = roe_std / roe_mean if roe_mean > 0 else float('inf')
                    
                    if roe_cv < 0.1:  # Very stable
                        score += 2
                        details.append(f"Extremely stable ROE (CV: {roe_cv:.2f})")
                    elif roe_cv < 0.2:  # Stable
                        score += 1
                        details.append(f"Stable ROE (CV: {roe_cv:.2f})")
                    else:
                        details.append(f"Variable ROE (CV: {roe_cv:.2f})")
                except Exception:
                    details.append("Unable to calculate ROE stability")
        else:
            details.append("Insufficient ROE history for stability analysis")
    else:
        details.append("ROE history data not available")
    
    # 2. Check for stable and high operating margins over time
    margin_data = get_operating_margin(ticker)
    if margin_data and "historical_margins" in margin_data and margin_data["historical_margins"]:
        historical_margins = margin_data["historical_margins"]
        
        if len(historical_margins) >= 3:
            # Count number of periods with operating margin > 15%
            high_margin_periods = sum(1 for margin in historical_margins if margin > 0.15)
            
            if high_margin_periods == len(historical_margins):
                score += 3
                details.append(f"Consistently high operating margins >15% across all {len(historical_margins)} periods")
            elif high_margin_periods >= len(historical_margins) * 0.7:
                score += 2
                details.append(f"Strong operating margins >15% in {high_margin_periods}/{len(historical_margins)} periods")
            elif high_margin_periods > 0:
                score += 1
                details.append(f"Occasional high operating margins >15% in {high_margin_periods}/{len(historical_margins)} periods")
            else:
                details.append("No periods with operating margin >15%")
            
            # Also check margin stability (variation)
            if len(historical_margins) >= 4:
                try:
                    margin_std = statistics.stdev(historical_margins)
                    margin_mean = statistics.mean(historical_margins)
                    margin_cv = margin_std / margin_mean if margin_mean > 0 else float('inf')
                    
                    if margin_cv < 0.1:  # Very stable
                        score += 2
                        details.append(f"Extremely stable operating margins (CV: {margin_cv:.2f})")
                    elif margin_cv < 0.2:  # Stable
                        score += 1
                        details.append(f"Stable operating margins (CV: {margin_cv:.2f})")
                    else:
                        details.append(f"Variable operating margins (CV: {margin_cv:.2f})")
                except Exception:
                    details.append("Unable to calculate margin stability")
        else:
            details.append("Insufficient margin history for stability analysis")
    else:
        details.append("Operating margin history data not available")
    
    # Normalize score to 0-10 scale
    max_raw_score = 10  # Maximum score from all criteria
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
    summary = f"Moat Analysis: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Strong evidence of durable competitive advantages."
    elif final_score >= 5:
        summary += ". Evidence of moderate competitive advantages."
    elif final_score >= 2.5:
        summary += ". Limited competitive advantages."
    else:
        summary += ". Little to no evidence of competitive advantages."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary,
        "max_score": 10
    }

def analyze_management_quality(ticker: str) -> Dict[str, Any]:
    """
    Evaluate management quality based on Buffett's criteria:
    - Capital allocation decisions
    - Share repurchases at good prices
    - Avoiding dilution
    - Rational dividend policy
    - Reinvestment efficiency
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0, 
            "details": "Insufficient data for management analysis", 
            "rating": "Poor",
            "summary": "Management Quality: Poor (0/10 points). Insufficient data.",
            "max_score": 10
        }
    
    # 1. Check capital allocation using ROIC or ROE trends
    roe_data = get_roe(ticker)
    if roe_data and "historical_data" in roe_data and roe_data["historical_data"]:
        historical_roes = roe_data["historical_data"]
        
        if len(historical_roes) >= 3:
            # Check if ROE is stable or improving over time
            improving_roe = all(historical_roes[i] >= historical_roes[i+1] for i in range(len(historical_roes)-1))
            
            if improving_roe:
                score += 2
                details.append("Improving ROE trend indicates good capital allocation")
            elif sum(historical_roes) / len(historical_roes) > 0.15:
                score += 1
                details.append("Consistently good ROE indicates reasonable capital allocation")
            else:
                details.append("Weak or declining ROE suggests poor capital allocation")
        else:
            details.append("Insufficient ROE history for capital allocation analysis")
    else:
        details.append("ROE history data not available")
    
    # 2. Check for share repurchases/dilution
    cash_flows = fundamental_data.get("cash_flows", {})
    share_repurchases = []
    
    for repurchase_key in ["Share Repurchase", "Repurchase of Stock", "Common Stock Repurchased"]:
        if repurchase_key in cash_flows:
            share_repurchases = cash_flows[repurchase_key]
            share_repurchases = [float(sr) if sr is not None else None for sr in share_repurchases]
            break
    
    # Look for outstanding shares data
    balance_sheet = fundamental_data.get("balance_sheet", {})
    outstanding_shares = []
    
    for share_key in ["Common Shares Outstanding", "Shares Outstanding", "Total Common Shares Outstanding"]:
        if share_key in balance_sheet:
            outstanding_shares = balance_sheet[share_key]
            outstanding_shares = [float(os) if os is not None else None for os in outstanding_shares]
            break
    
    # Analyze share count trends if available
    if outstanding_shares and len([s for s in outstanding_shares if s is not None]) >= 3:
        valid_shares = [s for s in outstanding_shares if s is not None]
        
        # Check if share count is decreasing (buybacks) or increasing (dilution)
        if valid_shares[0] < valid_shares[-1]:  # Fewer shares now than before
            score += 2
            reduction_pct = (valid_shares[-1] - valid_shares[0]) / valid_shares[-1]
            details.append(f"Reduced share count by {reduction_pct:.1%} over time (buybacks)")
        elif valid_shares[0] > valid_shares[-1] * 1.1:  # More than 10% dilution
            details.append(f"Significant share dilution: {(valid_shares[0] / valid_shares[-1] - 1):.1%} increase")
        else:
            score += 1
            details.append("Stable share count indicates no significant dilution")
    elif share_repurchases and any(sr is not None and sr < 0 for sr in share_repurchases):
        # Some evidence of share repurchases
        score += 1
        details.append("Evidence of share repurchases in some periods")
    else:
        details.append("Share count/repurchase data not available")
    
    # 3. Check dividend policy
    dividend_values = []
    for dividend_key in ["Dividend Paid", "Dividends Paid", "Common Stock Dividend Paid"]:
        if dividend_key in cash_flows:
            dividend_values = cash_flows[dividend_key]
            dividend_values = [float(d) if d is not None else None for d in dividend_values]
            break
    
    profit_loss = fundamental_data.get("profit_loss", {})
    net_income_values = []
    for income_key in ["Net Income", "Net Profit", "Profit After Tax"]:
        if income_key in profit_loss:
            net_income_values = profit_loss[income_key]
            net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
            break
    
    # Calculate dividend payout ratio if both metrics available
    if dividend_values and net_income_values and len(dividend_values) >= 3 and len(net_income_values) >= 3:
        # Use the minimum length
        min_length = min(len(dividend_values), len(net_income_values))
        payout_ratios = []
        
        for i in range(min_length):
            if (dividend_values[i] is not None and dividend_values[i] < 0 and 
                net_income_values[i] is not None and net_income_values[i] > 0):
                # Dividend values are typically negative in cash flow statements
                payout_ratio = abs(dividend_values[i]) / net_income_values[i]
                payout_ratios.append(payout_ratio)
        
        if payout_ratios:
            avg_payout = sum(payout_ratios) / len(payout_ratios)
            
            if 0.2 <= avg_payout <= 0.5:  # Rational payout ratio
                score += 2
                details.append(f"Prudent dividend policy with {avg_payout:.1%} average payout ratio")
            elif avg_payout > 0:
                score += 1
                details.append(f"Pays dividends with {avg_payout:.1%} average payout ratio")
            else:
                details.append("No dividend payments detected")
        else:
            details.append("Unable to calculate dividend payout ratios")
    else:
        details.append("Insufficient data for dividend policy analysis")
    
    # 4. Check reinvestment efficiency
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "historical_fcf" in fcf_data and fcf_data["historical_fcf"]:
        historical_fcf = fcf_data["historical_fcf"]
        
        if len(historical_fcf) >= 3:
            # Check FCF growth pattern
            fcf_growth_periods = 0
            for i in range(len(historical_fcf) - 1):
                if historical_fcf[i] > historical_fcf[i + 1]:  # Growing FCF
                    fcf_growth_periods += 1
            
            fcf_growth_ratio = fcf_growth_periods / (len(historical_fcf) - 1)
            
            if fcf_growth_ratio >= 0.7:  # Consistent growth in 70%+ of periods
                score += 2
                details.append(f"Efficient reinvestment: Growing FCF in {fcf_growth_periods}/{len(historical_fcf)-1} periods")
            elif fcf_growth_ratio >= 0.5:  # Growth in majority of periods
                score += 1
                details.append(f"Moderate reinvestment efficiency: Growing FCF in {fcf_growth_periods}/{len(historical_fcf)-1} periods")
            else:
                details.append(f"Poor reinvestment: Growing FCF in only {fcf_growth_periods}/{len(historical_fcf)-1} periods")
        else:
            details.append("Insufficient FCF history for reinvestment analysis")
    else:
        details.append("FCF history data not available")
    
    # 5. Additional consideration: debt management
    debt_equity_data = get_debt_to_equity(ticker)
    if debt_equity_data and "historical_ratios" in debt_equity_data and debt_equity_data["historical_ratios"]:
        historical_debt_equity = debt_equity_data["historical_ratios"]
        
        if len(historical_debt_equity) >= 3:
            # Check if debt levels are decreasing over time
            decreasing_debt = all(historical_debt_equity[i] <= historical_debt_equity[i+1] for i in range(len(historical_debt_equity)-1))
            
            if decreasing_debt:
                score += 2
                details.append("Prudent debt reduction over time")
            elif max(historical_debt_equity) < 0.5:
                score += 1
                details.append("Consistently low debt levels")
            elif max(historical_debt_equity) > 2.0:
                details.append("Concerning high debt levels")
            else:
                details.append("Stable moderate debt levels")
        else:
            details.append("Insufficient debt history for trend analysis")
    else:
        details.append("Debt ratio history not available")
    
    # Normalize score to 0-10 scale
    max_raw_score = 10  # Maximum score from all criteria
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
    summary = f"Management Quality: {rating} ({final_score:.1f}/10 points)"
    if final_score >= 7.5:
        summary += ". Exceptional capital allocation with shareholder-friendly policies."
    elif final_score >= 5:
        summary += ". Good management decisions with reasonable capital allocation."
    elif final_score >= 2.5:
        summary += ". Average management quality with some questionable decisions."
    else:
        summary += ". Poor management decisions and capital allocation."
    
    return {
        "score": final_score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary,
        "max_score": 10
    }

def calculate_owner_earnings(ticker: str) -> Dict[str, Any]:
    """
    Calculate owner earnings using Buffett's preferred formula:
    Owner Earnings = Net Income + Depreciation/Amortization - Maintenance CapEx
    
    Since maintenance CapEx is hard to determine, we use a heuristic:
    Either use full CapEx as a conservative estimate or use a fraction of total CapEx.
    """
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "owner_earnings": None,
            "details": "Insufficient data for owner earnings calculation",
            "owner_earnings_estimate": None
        }
    
    profit_loss = fundamental_data.get("profit_loss", {})
    cash_flows = fundamental_data.get("cash_flows", {})
    
    # Get net income values
    net_income_values = []
    for income_key in ["Net Income", "Net Profit", "Profit After Tax"]:
        if income_key in profit_loss:
            net_income_values = profit_loss[income_key]
            net_income_values = [float(ni) if ni is not None else None for ni in net_income_values]
            break
    
    # Get depreciation/amortization values
    depreciation_values = []
    for dep_key in ["Depreciation and Amortization", "Depreciation", "Depreciation & Amortization"]:
        if dep_key in cash_flows:
            depreciation_values = cash_flows[dep_key]
            depreciation_values = [float(d) if d is not None else None for d in depreciation_values]
            break
    
    # Get capital expenditure values
    capex_values = []
    for capex_key in ["Capital Expenditure", "CAPEX", "Purchase of Fixed Assets"]:
        if capex_key in cash_flows:
            capex_values = cash_flows[capex_key]
            capex_values = [float(c) if c is not None else None for c in capex_values]
            break
    
    # Calculate owner earnings
    if net_income_values and depreciation_values and capex_values:
        min_length = min(len(net_income_values), len(depreciation_values), len(capex_values))
        owner_earnings_values = []
        
        for i in range(min_length):
            if (net_income_values[i] is not None and 
                depreciation_values[i] is not None and 
                capex_values[i] is not None):
                # Note: Depreciation usually positive in cash flow, CapEx negative
                # Maintenance CapEx is assumed to be a fraction (70%) of total CapEx as a rough estimate
                maintenance_capex = abs(capex_values[i]) * 0.7
                owner_earnings = net_income_values[i] + abs(depreciation_values[i]) - maintenance_capex
                owner_earnings_values.append(owner_earnings)
        
        if owner_earnings_values:
            avg_owner_earnings = sum(owner_earnings_values) / len(owner_earnings_values)
            
            # Return both average and most recent
            return {
                "owner_earnings": owner_earnings_values[0] if owner_earnings_values else None,
                "details": f"Average owner earnings over {len(owner_earnings_values)} periods: {avg_owner_earnings:.2f}",
                "owner_earnings_estimate": avg_owner_earnings,
                "historical_values": owner_earnings_values
            }
    
    # Use FCF as a fallback if owner earnings can't be calculated
    fcf_data = get_free_cash_flow(ticker)
    if fcf_data and "historical_fcf" in fcf_data and fcf_data["historical_fcf"]:
        historical_fcf = fcf_data["historical_fcf"]
        
        if historical_fcf:
            avg_fcf = sum(historical_fcf) / len(historical_fcf)
            return {
                "owner_earnings": None,
                "details": f"Using FCF as proxy for owner earnings: {avg_fcf:.2f} (average over {len(historical_fcf)} periods)",
                "owner_earnings_estimate": avg_fcf,
                "historical_values": historical_fcf
            }
    
    return {
        "owner_earnings": None,
        "details": "Insufficient data for owner earnings or FCF calculation",
        "owner_earnings_estimate": None
    }


def calculate_intrinsic_value(ticker: str, market_cap: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate intrinsic value using Buffett's preferred DCF method.
    Buffett typically uses owner earnings and applies a discount rate
    to determine present value.
    """
    score = 0
    details = []
    
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
                "summary": "Valuation: Unknown (0/10 points). Insufficient market data.",
                "intrinsic_value": None,
                "margin_of_safety": None
            }
    
    # Calculate owner earnings
    owner_earnings_data = calculate_owner_earnings(ticker)
    owner_earnings = owner_earnings_data["owner_earnings_estimate"]
    
    if not owner_earnings:
        return {
            "score": 0,
            "details": "Unable to calculate owner earnings or FCF",
            "rating": "Unknown",
            "summary": "Valuation: Unknown (0/10 points). Insufficient data for intrinsic value calculation.",
            "intrinsic_value": None,
            "margin_of_safety": None
        }
    
    # Use the constant growth perpetuity model for intrinsic value
    # IV = E / (r - g)
    # where E = owner earnings, r = discount rate, g = growth rate
    
    # Buffett uses relatively conservative parameters
    discount_rate = 0.10  # 10% discount rate
    
    # Estimate growth rate based on historical data
    historical_values = owner_earnings_data.get("historical_values", [])
    growth_rate = 0.03  # Default assumption of 3% perpetual growth
    
    if len(historical_values) >= 3:
        # Calculate average growth rate if possible
        growth_rates = []
        for i in range(len(historical_values) - 1):
            if historical_values[i+1] > 0 and historical_values[i] > 0:
                # Note: index 0 is latest period in our data
                annual_growth = (historical_values[i] / historical_values[i+1]) - 1
                growth_rates.append(annual_growth)
        
        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            # Cap growth rate between 0% and 6% for conservatism
            growth_rate = max(0, min(0.06, avg_growth))
            details.append(f"Using historical growth rate of {growth_rate:.1%}")
        else:
            details.append(f"Using default growth rate of {growth_rate:.1%}")
    else:
        details.append(f"Using default growth rate of {growth_rate:.1%}")
    
    # Calculate intrinsic value
    if discount_rate > growth_rate:  # Ensure formula works mathematically
        intrinsic_value = owner_earnings / (discount_rate - growth_rate)
        
        # Calculate margin of safety
        margin_of_safety = (intrinsic_value - market_cap) / market_cap
        
        # Determine score based on margin of safety
        if margin_of_safety >= 0.3:  # 30% or more discount to intrinsic value
            score = 10
            details.append(f"Significant margin of safety: {margin_of_safety:.1%}")
        elif margin_of_safety >= 0.1:  # 10-30% discount
            score = 7.5
            details.append(f"Moderate margin of safety: {margin_of_safety:.1%}")
        elif margin_of_safety >= -0.1:  # Within 10% of fair value
            score = 5
            details.append(f"Limited margin of safety: {margin_of_safety:.1%}")
        elif margin_of_safety >= -0.3:  # 10-30% premium
            score = 2.5
            details.append(f"Overvalued by {-margin_of_safety:.1%}")
        else:  # More than 30% premium
            score = 0
            details.append(f"Significantly overvalued by {-margin_of_safety:.1%}")
        
        # Also check earnings yield as a secondary metric
        earnings_yield = owner_earnings / market_cap
        if earnings_yield >= 0.10:  # 10% or higher yield
            details.append(f"Attractive earnings yield of {earnings_yield:.1%}")
        else:
            details.append(f"Earnings yield of {earnings_yield:.1%}")
    else:
        # If discount rate <= growth rate, the perpetuity formula doesn't work
        intrinsic_value = None
        margin_of_safety = None
        score = 0
        details.append("Unable to calculate intrinsic value: growth rate exceeds discount rate")
    
    # Determine rating based on score
    if score >= 7.5:
        rating = "Undervalued"
    elif score >= 5:
        rating = "Fairly Valued"
    elif score >= 2.5:
        rating = "Slightly Overvalued"
    else:
        rating = "Overvalued"
    
    # Generate summary
    summary = f"Valuation: {rating} ({score:.1f}/10 points)"
    if intrinsic_value:
        summary += f". Intrinsic value estimate: {intrinsic_value:.0f} vs Market Cap: {market_cap:.0f}"
        if margin_of_safety > 0:
            summary += f", representing a {margin_of_safety:.1%} margin of safety."
        else:
            summary += f", representing a {-margin_of_safety:.1%} premium to fair value."
    else:
        summary += ". Unable to calculate reliable intrinsic value."
    
    return {
        "score": score,
        "details": "; ".join(details),
        "rating": rating,
        "summary": summary,
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
        "owner_earnings": owner_earnings
    } 

class WarrenBuffettAgnoAgent():
    """Agno-based agent implementing Warren Buffett's investment principles."""
    
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
        prompt_template = """You are a Warren Buffett AI agent, making investment decisions using his principles:

            1. Focus on companies with strong fundamentals and consistent performance.
            2. Look for businesses with durable competitive advantages ("moats").
            3. Place high value on good management that allocates capital effectively.
            4. Invest with a margin of safety by buying below intrinsic value.
            5. Be patient and think long-term rather than focusing on short-term fluctuations.
            6. Avoid businesses you don't understand and stay within your circle of competence.
            7. Be fearful when others are greedy, and greedy when others are fearful.
            
            Please analyze {ticker} using the following data:

            Fundamental Analysis:
            {fundamental_analysis}

            Consistency Analysis:
            {consistency_analysis}

            Moat Analysis:
            {moat_analysis}

            Management Quality Analysis:
            {management_analysis}

            Intrinsic Value Analysis:
            {intrinsic_value_analysis}

            Based on this analysis and Warren Buffett's investment philosophy, would you recommend investing in {ticker}?
            Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
            Explain your reasoning in Buffett's characteristic style, focusing on the long-term prospects and margin of safety.
            """
        return prompt_template.format(
            ticker=ticker,
            fundamental_analysis=json.dumps(analysis_summary["fundamental_analysis"], indent=2),
            consistency_analysis=json.dumps(analysis_summary["consistency_analysis"], indent=2),
            moat_analysis=json.dumps(analysis_summary["moat_analysis"], indent=2),
            management_analysis=json.dumps(analysis_summary["management_analysis"], indent=2),
            intrinsic_value_analysis=json.dumps(analysis_summary["intrinsic_value_analysis"], indent=2)
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Warren Buffett's principles.
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "warren_buffett_agno_agent"

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
                
                # Get market cap - handle both dictionary and direct float return
                market_cap_data = get_market_cap(ticker)
                market_cap = None
                if isinstance(market_cap_data, dict) and "current_market_cap" in market_cap_data:
                    market_cap = market_cap_data["current_market_cap"]
                elif isinstance(market_cap_data, (int, float)):
                    market_cap = market_cap_data
                
                # Perform analysis using the direct API
                fundamental_analysis = analyze_fundamentals(ticker)
                consistency_analysis = analyze_consistency(ticker)
                moat_analysis = analyze_moat(ticker)
                management_analysis = analyze_management_quality(ticker)
                intrinsic_value_analysis = calculate_intrinsic_value(ticker, market_cap)
                
                # Calculate total score with Buffett's weighting preferences
                total_score = (
                    fundamental_analysis.get("score", 0) * 0.25 +
                    consistency_analysis.get("score", 0) * 0.15 +
                    moat_analysis.get("score", 0) * 0.25 +
                    management_analysis.get("score", 0) * 0.15 +
                    intrinsic_value_analysis.get("score", 0) * 0.20
                )
                
                # Calculate maximum possible score
                max_score = 10.0  # All components normalized to 0-10 scale
                
                # Determine margin of safety
                margin_of_safety = intrinsic_value_analysis.get("margin_of_safety")
                
                # Generate trading signal directly
                # if fundamentals+moat+management are strong but margin_of_safety < 0.3, it's neutral
                # if fundamentals are weak or margin_of_safety is severely negative -> bearish
                # else bullish
                if (total_score >= 0.7 * max_score) and margin_of_safety and (margin_of_safety >= 0.3):
                    signal = "bullish"
                elif total_score <= 0.3 * max_score or (margin_of_safety is not None and margin_of_safety < -0.3):
                    # Negative margin of safety beyond -30% could be overpriced -> bearish
                    signal = "bearish"
                else:
                    signal = "neutral"
                
                analysis = {
                    "fundamental_analysis": fundamental_analysis,
                    "consistency_analysis": consistency_analysis,
                    "moat_analysis": moat_analysis,
                    "management_analysis": management_analysis,
                    "intrinsic_value_analysis": intrinsic_value_analysis,
                    "signal": signal,
                    "score": total_score,
                    "max_score": max_score,
                    "market_cap": market_cap
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
    
    def _parse_response(self, response: str, ticker: str) -> WarrenBuffettSignal:
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
            
            return WarrenBuffettSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
            return WarrenBuffettSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )

    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Warren Buffett's investment criteria"""
        print(f"Starting Warren Buffett-based analysis for {ticker}")
        
        # Get market cap - handle both dictionary and direct float return
        market_cap_data = get_market_cap(ticker)
        market_cap = None
        if isinstance(market_cap_data, dict) and "current_market_cap" in market_cap_data:
            market_cap = market_cap_data["current_market_cap"]
        elif isinstance(market_cap_data, (int, float)):
            market_cap = market_cap_data
        
        # Analyze the five key aspects of the investment
        fundamental_analysis = analyze_fundamentals(ticker)
        consistency_analysis = analyze_consistency(ticker)
        moat_analysis = analyze_moat(ticker)
        management_analysis = analyze_management_quality(ticker)
        intrinsic_value_analysis = calculate_intrinsic_value(ticker, market_cap)
        
        # Calculate overall score with Buffett's weighting preferences
        total_score = (
            fundamental_analysis.get("score", 0) * 0.25 +
            consistency_analysis.get("score", 0) * 0.15 +
            moat_analysis.get("score", 0) * 0.25 +
            management_analysis.get("score", 0) * 0.15 +
            intrinsic_value_analysis.get("score", 0) * 0.20
        )
        
        # Determine investment recommendation
        if total_score >= 7.5:
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
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "overall_score": total_score,
            "recommendation": recommendation,
            "summary": f"""
Warren Buffett Analysis for {ticker}:

{fundamental_analysis['summary']}

{consistency_analysis['summary']}

{moat_analysis['summary']}

{management_analysis['summary']}

{intrinsic_value_analysis['summary']}

Overall Score: {total_score:.1f}/10
Recommendation: {recommendation}
"""
        }
        
        print(f"Completed Warren Buffett-based analysis for {ticker}")
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
        agent = WarrenBuffettAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 