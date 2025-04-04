"""
Cathie Wood Investing Agent using Agno Framework
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
from tools.fundamental import fetch_fundamental_data, get_market_cap, get_revenue_growth
from tools.fundamental import get_operating_margin, get_free_cash_flow, get_research_and_development
from tools.fundamental import get_gross_margin, get_net_income, get_depreciation_amortization
from tools.fundamental import get_capital_expenditure, get_working_capital_change
from tools.ohlcv import fetch_ohlcv_data, OHLCVData

def safe_float_convert(value):
    """
    Safely convert a value to float, returning None if conversion fails.
    
    Args:
        value: The value to convert to float
        
    Returns:
        float value if conversion is successful, None otherwise
    """
    if value is None:
        return None
    
    try:
        # Try to convert to float
        if isinstance(value, str):
            # Remove any percentage or currency symbols or commas
            cleaned = value.replace('%', '').replace('$', '').replace(',', '').strip()
            return float(cleaned)
        return float(value)
    except (ValueError, TypeError):
        return None

# Pydantic model for the output signal
class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def analyze_disruptive_potential(ticker: str) -> dict:
    """
    Analyze whether the company has disruptive products, technology, or business model.
    Evaluates multiple dimensions of disruptive potential:
    1. Revenue Growth Acceleration - indicates market adoption
    2. Gross Margin Trends - suggests pricing power and scalability
    3. Operating Leverage - demonstrates business model efficiency
    4. Market Share Dynamics - indicates competitive position
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0,
            "details": "Insufficient data to analyze disruptive potential"
        }
    
    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenue_data = get_revenue_growth(ticker)
    if revenue_data and "growth_rate" in revenue_data:
        growth_rate = revenue_data["growth_rate"]
        
        # Check absolute growth rate
        if growth_rate > 0.3:  # 30% growth
            score += 3
            details.append(f"Exceptional revenue growth: {(growth_rate*100):.1f}%")
        elif growth_rate > 0.15:  # 15% growth
            score += 2
            details.append(f"Strong revenue growth: {(growth_rate*100):.1f}%")
        elif growth_rate > 0.07:  # 7% growth
            score += 1
            details.append(f"Moderate revenue growth: {(growth_rate*100):.1f}%")
        
        # Check if growth data indicates disruptive potential
        if "source" in revenue_data and revenue_data["source"] == "compounded_sales_growth":
            score += 1
            details.append("Consistent compound growth suggests strong market position")
    else:
        details.append("Revenue growth data not available")
    
    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margin_data = get_gross_margin(ticker)
    
    if gross_margin_data and "values" in gross_margin_data and len(gross_margin_data["values"]) >= 2:
        gross_margins = gross_margin_data["values"]
        
        # Check for trend
        margin_trend = gross_margins[0] - gross_margins[-1]  # current - oldest
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level
        current_margin = gross_margin_data.get("current_value")
        if current_margin:
            if current_margin > 0.50:  # High margin business
                score += 2
                details.append(f"High gross margin: {(current_margin*100):.1f}%")
            elif current_margin > 0.35:  # Good margin business
                score += 1
                details.append(f"Good gross margin: {(current_margin*100):.1f}%")
            
            if "is_estimated" in gross_margin_data and gross_margin_data["is_estimated"]:
                details.append("Note: Gross margin is estimated")
    else:
        details.append("Insufficient gross margin data")
    
    # 3. Operating Leverage Analysis
    operating_data = get_operating_margin(ticker)
    revenue_data = get_revenue_growth(ticker)
    
    if operating_data and "current_operating_margin" in operating_data:
        current_margin = operating_data["current_operating_margin"]
        if current_margin > 0.2:  # Strong operating margin
            score += 2
            details.append(f"Strong operating margin: {(current_margin*100):.1f}%")
        elif current_margin > 0.1:  # Decent operating margin
            score += 1
            details.append(f"Decent operating margin: {(current_margin*100):.1f}%")
            
        # Check for trend if available
        if "trend" in operating_data:
            if operating_data["trend"] == "improving":
                score += 1
                details.append("Improving operating margin trend")
    else:
        details.append("Operating margin data not available")
    
    # 4. Market position assessment based on profit growth vs industry
    if "compounded_profit_growth" in fundamental_data:
        profit_growth = fundamental_data["compounded_profit_growth"]
        if "Compounded Profit Growth" in profit_growth:
            growth_values = profit_growth["Compounded Profit Growth"]
            if growth_values and len(growth_values) > 0:
                growth_rate = safe_float_convert(growth_values[0])
                if growth_rate:
                    # Convert percentage to decimal if needed
                    if growth_rate > 1:
                        growth_rate = growth_rate / 100
                        
                    if growth_rate > 0.25:  # Very high profit growth
                        score += 3
                        details.append(f"Exceptional profit growth rate: {(growth_rate*100):.1f}%")
                    elif growth_rate > 0.15:
                        score += 2
                        details.append(f"Strong profit growth rate: {(growth_rate*100):.1f}%")
                    elif growth_rate > 0.10:
                        score += 1
                        details.append(f"Good profit growth rate: {(growth_rate*100):.1f}%")
    
    # Normalize score to be out of 5
    max_possible_score = 13  # Sum of all possible points
    normalized_score = min(5, (score / max_possible_score) * 5)
    
    # Create rating
    rating = "Excellent" if normalized_score >= 4 else "Good" if normalized_score >= 3 else "Average" if normalized_score >= 2 else "Poor"
    
    # Generate summary
    summary = f"Disruptive Potential: {rating} ({normalized_score:.1f}/5 points)"
    if normalized_score >= 4:
        summary += ". Strong evidence of disruptive innovation and market adoption."
    elif normalized_score >= 3:
        summary += ". Good indicators of innovation with competitive advantages."
    elif normalized_score >= 2:
        summary += ". Some innovative elements but limited disruption potential."
    else:
        summary += ". Limited innovation metrics by Wood's criteria."

    return {
        "score": normalized_score,
        "rating": rating,
        "details": details,
        "summary": summary,
        "raw_score": score,
        "max_score": max_possible_score
    }

def analyze_innovation_growth(ticker: str) -> dict:
    """
    Evaluate the company's potential for exponential growth.
    Analyzes multiple dimensions:
    1. Sales and Profit Growth Trends
    2. Free Cash Flow Generation - indicates ability to fund innovation
    3. Operating Efficiency - shows scalability
    4. Capital Allocation - reveals growth-focused management
    5. Growth Reinvestment - demonstrates commitment to future growth
    """
    score = 0
    details = []
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {
            "score": 0,
            "details": "Insufficient data to analyze innovation-driven growth"
        }
    
    # 1. Sales and Profit Growth Analysis
    revenue_data = get_revenue_growth(ticker)
    
    if revenue_data and "growth_rate" in revenue_data:
        growth_rate = revenue_data["growth_rate"]
        
        # Check absolute growth rate
        if growth_rate > 0.3:  # 30% growth
            score += 3
            details.append(f"Exceptional revenue growth: {(growth_rate*100):.1f}%")
        elif growth_rate > 0.15:  # 15% growth
            score += 2
            details.append(f"Strong revenue growth: {(growth_rate*100):.1f}%")
        elif growth_rate > 0.07:  # 7% growth
            score += 1
            details.append(f"Moderate revenue growth: {(growth_rate*100):.1f}%")
    else:
        details.append("Revenue growth data not available")
    
    # Check profit growth data separately for comparison with revenue growth
    if "compounded_profit_growth" in fundamental_data:
        profit_growth = fundamental_data["compounded_profit_growth"]
        if "Compounded Profit Growth" in profit_growth:
            growth_values = profit_growth["Compounded Profit Growth"]
            if growth_values and len(growth_values) > 0:
                profit_growth_rate = safe_float_convert(growth_values[0])
                if profit_growth_rate:
                    # Convert percentage to decimal if needed
                    if profit_growth_rate > 1:
                        profit_growth_rate = profit_growth_rate / 100
                    
                    if profit_growth_rate > growth_rate:
                        score += 2
                        details.append(f"Profit growth ({profit_growth_rate*100:.1f}%) outpacing revenue growth, indicating scaling efficiency")
    
    # 2. Free Cash Flow Analysis
    fcf_data = get_free_cash_flow(ticker)
    
    # If FCF data is not available, try to calculate it manually
    if not fcf_data:
        ni_data = get_net_income(ticker)
        dep_data = get_depreciation_amortization(ticker)
        capex_data = get_capital_expenditure(ticker)
        wc_change_data = get_working_capital_change(ticker)
        
        if (ni_data and "current_value" in ni_data and
            dep_data and "current_value" in dep_data and
            capex_data and "current_value" in capex_data and
            wc_change_data and "current_change" in wc_change_data):
            
            # Calculate FCF manually
            ni = ni_data["current_value"]
            dep = dep_data["current_value"]
            capex = capex_data["current_value"]
            wc_change = wc_change_data["current_change"]
            
            manual_fcf = ni + dep - capex - wc_change
            
            # Get market cap for yield calculation
            market_cap = get_market_cap(ticker)
            fcf_yield = None
            if market_cap and market_cap > 0:
                fcf_yield = manual_fcf / market_cap
            
            # Create a simple FCF data structure
            fcf_data = {
                "current_fcf": manual_fcf,
                "fcf_yield": fcf_yield,
                "is_positive": manual_fcf > 0,
                "calculation_method": "manual"
            }
            
            details.append(f"FCF calculated manually: {manual_fcf:.1f}")
    
    if fcf_data and "current_fcf" in fcf_data:
        current_fcf = fcf_data["current_fcf"]
        is_positive = fcf_data.get("is_positive", False)

        if is_positive and current_fcf > 0:
            score += 2
            details.append("Strong positive FCF, excellent growth funding capacity")
            
            # Check FCF yield if available
            if "fcf_yield" in fcf_data and fcf_data["fcf_yield"]:
                fcf_yield = fcf_data["fcf_yield"]
                if fcf_yield > 0.05:  # 5% yield
                    score += 1
                    details.append(f"Excellent FCF yield: {fcf_yield*100:.1f}%")
                elif fcf_yield > 0.03:  # 3% yield
                    score += 0.5
                    details.append(f"Good FCF yield: {fcf_yield*100:.1f}%")
        elif is_positive:
            score += 1
            details.append("Positive FCF, adequate growth funding capacity")
    else:
        details.append("FCF data not available")
    
    # 3. Operating Efficiency Analysis
    operating_data = get_operating_margin(ticker)
    if operating_data and "current_operating_margin" in operating_data:
        op_margin = operating_data["current_operating_margin"]
        
        # Check margin level
        if op_margin > 0.2:  # Over 20%
            score += 3
            details.append(f"Exceptional operating margin: {op_margin*100:.1f}%")
        elif op_margin > 0.15:
            score += 2
            details.append(f"Strong operating margin: {op_margin*100:.1f}%")
        elif op_margin > 0.1:
            score += 1
            details.append(f"Good operating margin: {op_margin*100:.1f}%")
            
        # Check for trend if available
        if "is_consistently_high" in operating_data and operating_data["is_consistently_high"]:
            score += 1
            details.append("Consistently high operating margins indicate strong business model")
    else:
        details.append("Operating margin data not available")
    
    # 4. Capital Allocation Analysis
    capex_data = get_capital_expenditure(ticker)
    if capex_data and "current_value" in capex_data and revenue_data and "historical_values" in revenue_data:
        capex = capex_data["current_value"]
        revenues = revenue_data["historical_values"]
        
        if revenues and len(revenues) > 0:
            current_revenue = revenues[0]  # Most recent revenue
            capex_intensity = capex / current_revenue if current_revenue != 0 else 0
            
            if capex_intensity > 0.10:
                score += 2
                details.append(f"Strong investment in growth infrastructure: {capex_intensity*100:.1f}% of revenue")
            elif capex_intensity > 0.05:
                score += 1
                details.append(f"Moderate investment in growth infrastructure: {capex_intensity*100:.1f}% of revenue")
            else:
                details.append(f"Limited capital investment: {capex_intensity*100:.1f}% of revenue")
        else:
            details.append("Insufficient revenue data for CAPEX analysis")
    else:
        details.append("Insufficient CAPEX data")
    
    # 5. Growth Reinvestment Analysis
    dividends = []
    if "cash_flows" in fundamental_data and "Dividends Paid" in fundamental_data["cash_flows"]:
        dividends = [abs(d) for d in fundamental_data["cash_flows"]["Dividends Paid"] if d is not None]
    
    # Use either the calculated FCF or manual calculation
    if dividends and fcf_data and "current_fcf" in fcf_data:
        current_fcf = fcf_data["current_fcf"]
        # Only proceed if we have valid dividend data
        if dividends and len(dividends) > 0 and current_fcf != 0:
            latest_dividend = dividends[0]  # Most recent dividend
            latest_payout_ratio = latest_dividend / current_fcf if current_fcf != 0 else 1
            
            if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
                score += 2
                details.append(f"Strong focus on reinvestment over dividends: {latest_payout_ratio*100:.1f}% payout ratio")
            elif latest_payout_ratio < 0.4:
                score += 1
                details.append(f"Moderate focus on reinvestment over dividends: {latest_payout_ratio*100:.1f}% payout ratio")
            else:
                details.append(f"High dividend payout ratio: {latest_payout_ratio*100:.1f}%")
    else:
        details.append("Insufficient dividend data")
    
    # Normalize score to be out of 5
    max_possible_score = 16  # Sum of all possible points
    normalized_score = min(5, (score / max_possible_score) * 5)
    
    # Create rating
    rating = "Excellent" if normalized_score >= 4 else "Good" if normalized_score >= 3 else "Average" if normalized_score >= 2 else "Poor"
    
    # Generate summary
    summary = f"Innovation-Driven Growth: {rating} ({normalized_score:.1f}/5 points)"
    if normalized_score >= 4:
        summary += ". Excellent investment in future growth and innovation."
    elif normalized_score >= 3:
        summary += ". Good commitment to growth with strong reinvestment."
    elif normalized_score >= 2:
        summary += ". Some growth indicators but limited exponential growth potential."
    else:
        summary += ". Limited growth indicators by Wood's criteria."

    return {
        "score": normalized_score,
        "rating": rating,
        "details": details,
        "summary": summary,
        "raw_score": score,
        "max_score": max_possible_score
    }

def analyze_cathie_wood_valuation(ticker: str) -> dict:
    """
    Cathie Wood often focuses on long-term exponential growth potential. We use
    a simplified approach looking for a large total addressable market (TAM) and the
    company's ability to capture a sizable portion with higher future growth rates.
    """
    score = 0
    details = []
    
    # Get market cap and FCF data
    market_cap_data = get_market_cap(ticker)
    fcf_data = get_free_cash_flow(ticker)
    
    # If FCF data is not available, try to calculate it manually
    if not fcf_data:
        ni_data = get_net_income(ticker)
        dep_data = get_depreciation_amortization(ticker)
        capex_data = get_capital_expenditure(ticker)
        wc_change_data = get_working_capital_change(ticker)
        
        if (ni_data and "current_value" in ni_data and
            dep_data and "current_value" in dep_data and
            capex_data and "current_value" in capex_data and
            wc_change_data and "current_change" in wc_change_data):
            
            # Calculate FCF manually
            ni = ni_data["current_value"]
            dep = dep_data["current_value"]
            capex = capex_data["current_value"]
            wc_change = wc_change_data["current_change"]
            
            manual_fcf = ni + dep - capex - wc_change
            
            # Create a simple FCF data structure
            fcf_data = {
                "current_fcf": manual_fcf,
                "is_positive": manual_fcf > 0,
                "calculation_method": "manual"
            }
            
            details.append(f"FCF calculated manually: {manual_fcf:.1f}")
    
    if not market_cap_data:
        return {
            "score": 0,
            "rating": "Unknown",
            "details": ["Market capitalization data not available"],
            "summary": "Valuation: Unknown (0/5 points). Insufficient market data."
        }
        
    if not fcf_data or "current_fcf" not in fcf_data:
        return {
            "score": 0,
            "rating": "Unknown",
            "details": ["Insufficient data for valuation"],
            "summary": "Valuation: Unknown (0/5 points). Insufficient data."
        }
    
    market_cap = market_cap_data
    fcf = fcf_data["current_fcf"]
    
    if fcf <= 0:
        return {
            "score": 0,
            "rating": "Poor",
            "details": [f"No positive FCF for valuation; FCF = {fcf}"],
            "summary": "Valuation: Poor (0/5 points). Negative free cash flow is concerning for future growth."
        }
    
    # For disruptive innovators, we use more aggressive growth assumptions
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5
    
    # Calculate DCF with high growth
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value with higher multiple for disruptive companies
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) \
                    / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    
    # Calculate margin of safety
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    # Score based on margin of safety
    if margin_of_safety > 0.5:  # More than 50% upside
        score += 3
        details.append(f"Significant upside potential of {margin_of_safety:.1%}")
    elif margin_of_safety > 0.2:  # 20-50% upside
        score += 2
        details.append(f"Moderate upside potential of {margin_of_safety:.1%}")
    elif margin_of_safety > 0:  # 0-20% upside
        score += 1
        details.append(f"Limited upside potential of {margin_of_safety:.1%}")
    else:
        details.append(f"No margin of safety, potential downside of {-margin_of_safety:.1%}")
    
    # Also consider FCF yield for assessment
    fcf_yield = fcf / market_cap
    if fcf_yield > 0.05:  # Above 5%
        score += 2
        details.append(f"Strong current FCF yield of {fcf_yield:.1%}")
    elif fcf_yield > 0.02:  # Above 2%
        score += 1
        details.append(f"Moderate current FCF yield of {fcf_yield:.1%}")
    else:
        details.append(f"Low current FCF yield of {fcf_yield:.1%}")
    
    # Normalize score to be out of 5
    normalized_score = score
    
    # Create rating
    if normalized_score >= 4:
        rating = "Excellent"
    elif normalized_score >= 3:
        rating = "Good"
    elif normalized_score >= 2:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Generate summary
    valuation_description = "Undervalued" if margin_of_safety > 0.2 else \
                          "Fairly Valued" if margin_of_safety > -0.2 else \
                          "Overvalued"
    
    summary = f"Valuation: {rating} ({normalized_score}/5 points)"
    
    if normalized_score >= 4:
        summary += f". {valuation_description} with significant future growth potential."
    elif normalized_score >= 3:
        summary += f". {valuation_description} with good growth assumptions."
    elif normalized_score >= 2:
        summary += f". {valuation_description} with moderate growth potential."
    else:
        summary += f". {valuation_description} with limited growth expectations."
    
    return {
        "score": normalized_score,
        "rating": rating,
        "details": details,
        "summary": summary,
        "intrinsic_value": intrinsic_value,
        "market_cap": market_cap,
        "margin_of_safety": margin_of_safety
    }

class CathieWoodAgnoAgent():
    """Agno-based agent implementing Cathie Wood's disruptive innovation investment principles."""
    
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
        """
        prompt_template = """You are a Cathie Wood AI agent, making investment decisions using her principles:

                1. Seek companies leveraging disruptive innovation.
                2. Emphasize exponential growth potential, large TAM.
                3. Focus on technology, healthcare, or other future-facing sectors.
                4. Consider multi-year time horizons for potential breakthroughs.
                5. Accept higher volatility in pursuit of high returns.
                6. Evaluate management's vision and ability to invest in R&D.

                Please analyze {ticker} using the following data:

                Disruptive Potential Analysis:
                {disruptive_potential}

                Innovation-Driven Growth Analysis:
                {innovation_growth}

                Valuation Analysis:
                {valuation}

                Based on this analysis and Cathie Wood's investment philosophy, would you recommend investing in {ticker}?
                Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
                Structure your reasoning to follow Wood's disruptive innovation approach.
                """
        return prompt_template.format(
            ticker=ticker,
            disruptive_potential=json.dumps(analysis_summary["disruptive_potential"], indent=2),
            innovation_growth=json.dumps(analysis_summary["innovation_growth"], indent=2),
            valuation=json.dumps(analysis_summary["valuation"], indent=2)
        )
    
    def _parse_response(self, response: str, ticker: str) -> CathieWoodSignal:
        """Parse the LLM response into a structured signal."""
        try:
            # Check if response is a string or an Agno RunResponse object
            response_text = ""
            if hasattr(response, 'content'):
                # If it's an Agno RunResponse object
                response_text = response.content
            elif not isinstance(response, str):
                # Try to convert any other object to string
                response_text = str(response)
            else:
                response_text = response
            
            # Convert to lowercase for easier matching
            text_lower = response_text.lower()
            
            # Extract signal (looking for explicit mentions)
            signal = "neutral"  # Default
            if "bullish" in text_lower:
                signal = "bullish"
            elif "bearish" in text_lower:
                signal = "bearish"
                
            # Look for confidence mentions (as decimal or percentage)
            confidence = 0.5  # Default confidence
            
            # Try to find confidence as decimal (0.X format)
            decimal_matches = re.findall(r"confidence[:\s]+(\d+\.\d+)", text_lower)
            if decimal_matches and 0 <= float(decimal_matches[0]) <= 1:
                confidence = float(decimal_matches[0])
            
            # Try to find confidence as percentage (X% format)
            percentage_matches = re.findall(r"confidence[:\s]+(\d+)%", text_lower)
            if percentage_matches:
                confidence = float(percentage_matches[0]) / 100
                
            # As a fallback, try to find any number after "confidence"
            if confidence == 0.5:
                generic_matches = re.findall(r"confidence[:\s]+(\d+\.?\d*)", text_lower)
                if generic_matches:
                    val = float(generic_matches[0])
                    if val > 1:  # Assume it's a percentage if > 1
                        confidence = val / 100
                    else:
                        confidence = val
            
            # Cap confidence to valid range
            confidence = min(1.0, max(0.0, confidence))
            
            # Extract reasoning - anything after "reasoning" or the whole text if not found
            reasoning_pattern = r"(?:reasoning|analysis)(?::|is|:is)(.*?)(?:\n\n|\Z)"
            reasoning_matches = re.findall(reasoning_pattern, text_lower, re.DOTALL)
            
            if reasoning_matches:
                reasoning = reasoning_matches[0].strip()
            else:
                # If no specific reasoning section, use the whole response
                reasoning = response_text
                
            # Clean up reasoning
            reasoning = reasoning.strip()
            if len(reasoning) > 1000:
                reasoning = reasoning[:997] + "..."
                
            # If reasoning is too short, use more of the original response
            if len(reasoning) < 100 and len(response_text) > 100:
                reasoning = response_text[:997] + "..." if len(response_text) > 1000 else response_text
            
            return CathieWoodSignal(
                signal=signal,
                confidence=confidence,
                reasoning=reasoning
            )
        except Exception as e:
            # In case of any error, return a neutral signal
            print(f"Error parsing LLM response: {str(e)}")
            return CathieWoodSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )
    
    def _run_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Cathie Wood's disruptive innovation principles.
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "cathie_wood_agno_agent"

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
                print(f"Analyzing {ticker} with Cathie Wood's approach...")
                
                # Check for presence of key data sections
                fundamental_data = fetch_fundamental_data(ticker)
                if not fundamental_data:
                    results[ticker] = {"error": f"No fundamental data found for {ticker}"}
                    continue
                
                # Use the analyze method directly
                analysis_result = self.analyze(ticker)
                
                # Extract full analysis from the result
                disruptive_potential = analysis_result.get("disruptive_potential", {})
                innovation_growth = analysis_result.get("innovation_growth", {})
                valuation = analysis_result.get("valuation", {})
                
                # Extract signal information from the result
                signal = analysis_result.get("signal", "neutral")
                confidence = analysis_result.get("confidence", 50)
                reasoning = analysis_result.get("reasoning", "No reasoning provided")
                overall_score = analysis_result.get("overall_score", 0)
                
                # Create a full analysis dictionary
                analysis = {
                    "disruptive_potential": disruptive_potential,
                    "innovation_growth": innovation_growth,
                    "valuation": valuation
                }
                
                # Debug print the analysis
                print(f"Analysis for {ticker}:")
                print(json.dumps(analysis, indent=2))
                
                # Store results
                results[ticker] = {
                    "signal": signal,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "analysis": analysis,
                    "overall_score": overall_score
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {"error": f"Error analyzing {ticker}: {str(e)}"}
        
        return {agent_name: results}
    
    def run(self, ticker: str, end_date: str = None) -> tuple:
        """
        A simplified interface for running analysis on a single ticker.
        
        Args:
            ticker: The ticker symbol to analyze
            end_date: Optional end date for data (format: YYYY-MM-DD)
            
        Returns:
            Tuple of (signal, confidence, reasoning)
        """
        # Create a state object for the standard run method
        state = {
            "data": {
                "tickers": [ticker],
            },
            "metadata": {}
        }
        
        if end_date:
            state["data"]["end_date"] = end_date
        
        try:
            # Use the analyze method directly instead of extracting from _run_from_state results
            analysis_result = self.analyze(ticker)
            
            # Extract the needed values from the analysis result
            signal = analysis_result.get("signal", "neutral")
            confidence = analysis_result.get("confidence", 0)
            reasoning = analysis_result.get("reasoning", "No reasoning provided")
            
            return signal, confidence, reasoning
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            import traceback
            traceback.print_exc()
            return "neutral", 0, f"Error analyzing {ticker}: {str(e)}"
    
    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Cathie Wood's disruptive innovation criteria"""
        print(f"Starting Cathie Wood-based analysis for {ticker}")
        
        # Analyze the three key aspects of the investment
        disruptive_potential = analyze_disruptive_potential(ticker)
        innovation_growth = analyze_innovation_growth(ticker)
        valuation = analyze_cathie_wood_valuation(ticker)
        
        # Create a full analysis dictionary
        analysis_summary = {
            "disruptive_potential": disruptive_potential,
            "innovation_growth": innovation_growth,
            "valuation": valuation
        }
        
        # Prepare prompt and run LLM for nuanced analysis
        prompt = self._prepare_agent_prompt(ticker, analysis_summary)
        llm_response = self.agent.run(prompt)
        
        # Parse LLM response into structured signal
        signal_data = self._parse_response(llm_response, ticker)
        
        # Calculate overall score (weighted average) for reference
        disruptive_weight = 0.4  # 40% weight
        innovation_weight = 0.4  # 40% weight
        valuation_weight = 0.2   # 20% weight
        
        overall_score = (
            disruptive_potential.get("score", 0) * disruptive_weight +
            innovation_growth.get("score", 0) * innovation_weight +
            valuation.get("score", 0) * valuation_weight
        )
        
        # Generate comprehensive report
        report = {
            "ticker": ticker,
            "disruptive_potential": disruptive_potential,
            "innovation_growth": innovation_growth,
            "valuation": valuation,
            "overall_score": overall_score,
            "signal": signal_data.signal,
            "confidence": signal_data.confidence * 100,  # Convert to percentage
            "reasoning": signal_data.reasoning,
            "summary": f"""
Cathie Wood Analysis for {ticker}:

Disruptive Potential: {disruptive_potential.get('score', 0):.1f}/5
{disruptive_potential.get('summary', 'No summary available')}

Innovation-Driven Growth: {innovation_growth.get('score', 0):.1f}/5
{innovation_growth.get('summary', 'No summary available')}

Valuation: {valuation.get('score', 0):.1f}/5
{valuation.get('summary', 'No summary available')}

Overall Score: {overall_score:.1f}/5
Signal: {signal_data.signal.upper()} ({signal_data.confidence * 100:.1f}% confidence)
"""
        }
        
        print(f"Completed Cathie Wood-based analysis for {ticker}")
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
        agent = CathieWoodAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 