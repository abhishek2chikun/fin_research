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
    gross_margins = []
    if "profit_loss" in fundamental_data and "Gross Margin %" in fundamental_data["profit_loss"]:
        raw_margins = fundamental_data["profit_loss"]["Gross Margin %"]
        # Convert percentages to decimals if needed
        gross_margins = [m/100 if m and m > 1 else m for m in raw_margins if m is not None]
    
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[-1] - gross_margins[0]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level
        if gross_margins[-1] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[-1]*100):.1f}%")
        elif gross_margins[-1] > 0.35:  # Good margin business
            score += 1
            details.append(f"Good gross margin: {(gross_margins[-1]*100):.1f}%")
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
    capex = []
    if "cash_flows" in fundamental_data and "Capital Expenditure" in fundamental_data["cash_flows"]:
        capex = [abs(c) for c in fundamental_data["cash_flows"]["Capital Expenditure"] if c is not None]
    
    if capex and revenue_data and "historical_values" in revenue_data and len(capex) >= 2:
        revenues = revenue_data["historical_values"]
        capex_intensity = abs(capex[-1]) / revenues[-1] if revenues[-1] != 0 else 0
        capex_growth = (abs(capex[-1]) - abs(capex[0])) / abs(capex[0]) if capex[0] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")
    
    # 5. Growth Reinvestment Analysis
    dividends = []
    if "cash_flows" in fundamental_data and "Dividends Paid" in fundamental_data["cash_flows"]:
        dividends = [abs(d) for d in fundamental_data["cash_flows"]["Dividends Paid"] if d is not None]
    
    if dividends and fcf_data and "current_fcf" in fcf_data:
        current_fcf = fcf_data["current_fcf"]
        # Check if company prioritizes reinvestment over dividends
        latest_payout_ratio = dividends[-1] / current_fcf if current_fcf != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
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
    
    if not market_cap_data or not fcf_data or not fcf_data.get("current_fcf"):
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
                
                When providing your reasoning, be thorough and specific by:
                1. Identifying the specific disruptive technologies/innovations the company is leveraging
                2. Highlighting growth metrics that indicate exponential potential (revenue acceleration, expanding TAM)
                3. Discussing the long-term vision and transformative potential over 5+ year horizons
                4. Explaining how the company might disrupt traditional industries or create new markets
                5. Addressing R&D investment and innovation pipeline that could drive future growth
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
            
            return CathieWoodSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
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
                
                # Perform analysis using the modular functions
                disruptive_potential = analyze_disruptive_potential(ticker)
                innovation_growth = analyze_innovation_growth(ticker)
                valuation = analyze_cathie_wood_valuation(ticker)
                
                analysis = {
                    "disruptive_potential": disruptive_potential,
                    "innovation_growth": innovation_growth,
                    "valuation": valuation
                }
                
                # Debug print the analysis
                print(f"Analysis for {ticker}:")
                print(json.dumps(analysis, indent=2))
                
                # Calculate overall score using weighted approach
                disruptive_weight = 0.4
                innovation_weight = 0.4
                valuation_weight = 0.2
                
                overall_score = (
                    disruptive_potential.get("score", 0) * disruptive_weight +
                    innovation_growth.get("score", 0) * innovation_weight +
                    valuation.get("score", 0) * valuation_weight
                )
                
                # Determine base signal from overall score
                if overall_score >= 3.5:  # 70% of max score 5
                    base_signal = "bullish"
                    base_confidence = min(overall_score / 5, 1.0)
                elif overall_score <= 1.5:  # 30% of max score 5
                    base_signal = "bearish"
                    base_confidence = min((5 - overall_score) / 5, 1.0)
                else:
                    base_signal = "neutral"
                    base_confidence = 0.5
                
                # Prepare prompt and run LLM for nuanced analysis
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
            
        # Use the existing run method
        results = self._run_from_state(state)
        
        # Extract the result for the single ticker
        agent_name = "cathie_wood_agno_agent"
        ticker_result = results.get(agent_name, {}).get(ticker, {})
        
        if "error" in ticker_result:
            return "neutral", 0, ticker_result["error"]
            
        return (
            ticker_result.get("signal", "neutral"),
            ticker_result.get("confidence", 0) * 100,  # Convert to percentage
            ticker_result.get("reasoning", "No reasoning provided")
        )
    
    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Cathie Wood's disruptive innovation criteria"""
        print(f"Starting Cathie Wood-based analysis for {ticker}")
        
        # Analyze the three key aspects of the investment
        disruptive_potential = analyze_disruptive_potential(ticker)
        innovation_growth = analyze_innovation_growth(ticker)
        valuation = analyze_cathie_wood_valuation(ticker)
        
        # Calculate overall score (weighted average)
        disruptive_weight = 0.4  # 40% weight
        innovation_weight = 0.4  # 40% weight
        valuation_weight = 0.2   # 20% weight
        
        overall_score = (
            disruptive_potential.get("score", 0) * disruptive_weight +
            innovation_growth.get("score", 0) * innovation_weight +
            valuation.get("score", 0) * valuation_weight
        )
        
        # Determine investment recommendation
        if overall_score >= 4.0:
            recommendation = "Strong Buy"
        elif overall_score >= 3.0:
            recommendation = "Buy"
        elif overall_score >= 2.5:
            recommendation = "Hold"
        elif overall_score >= 2.0:
            recommendation = "Neutral"
        else:
            recommendation = "Avoid"
        
        # Generate comprehensive report
        report = {
            "ticker": ticker,
            "disruptive_potential": disruptive_potential,
            "innovation_growth": innovation_growth,
            "valuation": valuation,
            "overall_score": overall_score,
            "recommendation": recommendation,
            "summary": f"""
Cathie Wood Analysis for {ticker}:

Disruptive Potential: {disruptive_potential.get('score', 0):.1f}/5
{disruptive_potential.get('summary', 'No summary available')}

Innovation-Driven Growth: {innovation_growth.get('score', 0):.1f}/5
{innovation_growth.get('summary', 'No summary available')}

Valuation: {valuation.get('score', 0):.1f}/5
{valuation.get('summary', 'No summary available')}

Overall Score: {overall_score:.1f}/5
Recommendation: {recommendation}
"""
        }
        
        print(f"Completed Cathie Wood-based analysis for {ticker}")
        return report


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_state = {
        "data": {
            "tickers": ["TSLA"],  # Example ticker
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