"""
Bill Ackman Investing Agent using Agno Framework
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
from tools.fundamental import is_banking_entity, get_net_interest_margin, get_cost_to_income_ratio, get_bank_capital_adequacy
from tools.ohlcv import fetch_ohlcv_data, OHLCVData

# Import base financial agent

# Pydantic model for the output signal
class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def analyze_business_quality(ticker: str) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages, and potential for long-term growth.
    """
    score = 0
    details = []

    # 1. Multi-period revenue growth analysis using the revenue growth utility
    revenue_growth = get_revenue_growth(ticker)
    
    if revenue_growth and revenue_growth.get("growth_rate") is not None:
        growth_rate = revenue_growth["growth_rate"]
        growth_pct = growth_rate * 100
        
        if growth_rate > 0.5:  # 50% growth
            score += 2
            details.append(f"Revenue grew by {growth_pct:.1f}% over the full period.")
        elif growth_rate > 0:
            score += 1
            details.append(f"Revenue growth is positive but under 50% cumulatively ({growth_pct:.1f}%).")
        else:
            details.append(f"Revenue did not grow significantly ({growth_pct:.1f}%).")
    else:
        details.append("Revenue growth data not available or insufficient.")

    # 2. Operating margin analysis using the new utility function
    operating_margin_data = get_operating_margin(ticker)
    operating_margins_found = False
    
    if operating_margin_data and operating_margin_data.get("historical_margins"):
        operating_margins_found = True
        
        if operating_margin_data.get("is_consistently_high"):
            score += 2
            details.append("Operating margins have consistently exceeded 15%.")
        elif operating_margin_data.get("current_operating_margin") and operating_margin_data.get("current_operating_margin") >= 0.15:
            score += 1
            current_margin = operating_margin_data["current_operating_margin"] * 100
            details.append(f"Current operating margin is strong at {current_margin:.1f}%, but not consistently above 15% historically.")
        else:
            if operating_margin_data.get("current_operating_margin"):
                current_margin = operating_margin_data["current_operating_margin"] * 100
                details.append(f"Operating margin ({current_margin:.1f}%) is below Ackman's typical 15% threshold.")
            else:
                details.append("Operating margin not consistently above 15%.")
    
    if not operating_margins_found:
        details.append("No operating margin data across periods.")

    # 3. Free cash flow analysis using the new utility function
    fcf_data = get_free_cash_flow(ticker)
    
    if fcf_data and fcf_data.get("historical_fcf"):
        if fcf_data.get("is_positive"):
            score += 2
            details.append("Consistently positive free cash flow.")
            
            # Add FCF yield information if available
            if fcf_data.get("fcf_yield"):
                fcf_yield_pct = fcf_data["fcf_yield"] * 100
                if fcf_yield_pct > 5:
                    details.append(f"Strong free cash flow yield of {fcf_yield_pct:.1f}%.")
        else:
            details.append("Free cash flow is not consistently positive.")
    else:
        details.append("Free cash flow data not available.")

    # Overall business quality score
    quality_rating = "Excellent" if score >= 5 else "Good" if score >= 3 else "Average" if score >= 1 else "Poor"
    
    # Generate summary
    summary = f"Business Quality: {quality_rating} ({score}/6 points)"
    if score >= 5:
        summary += ". Strong revenue growth, margins, and cash flow generation."
    elif score >= 3:
        summary += ". Reasonable business quality with some strengths."
    elif score >= 1:
        summary += ". Limited business quality metrics."
    else:
        summary += ". Poor business quality by Ackman's criteria."
    
    return {
        "score": score,
        "rating": quality_rating,
        "details": details,
        "summary": summary
    }

def analyze_financial_discipline(ticker: str) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    - Return on equity consistency
    """
    score = 0
    details = []
    
    # 1. Debt to Equity Ratio trend using the new utility function
    debt_equity_data = get_debt_to_equity(ticker)
    
    if debt_equity_data and debt_equity_data.get("historical_ratios"):
        if debt_equity_data.get("is_low_leverage"):
            score += 2
            details.append("Debt to equity ratio consistently below 2.0.")
        elif debt_equity_data.get("current_ratio"):
            current_de = debt_equity_data["current_ratio"]
            if current_de > 3.0:
                details.append(f"High debt to equity ratio of {current_de:.2f}.")
            else:
                score += 1
                details.append(f"Moderate debt to equity level of {current_de:.2f}.")
    else:
        details.append("Debt to equity data not available.")
    
    # 2. Dividend payout / shareholder returns
    dividend_data = fetch_fundamental_data(ticker)
    if dividend_data and "profit_loss" in dividend_data and "Dividend Payout %" in dividend_data["profit_loss"]:
        dividend_payout = dividend_data["profit_loss"]["Dividend Payout %"]
        # Convert percentages to decimals if needed
        dividend_payout = [d/100 if d and d > 1 else d for d in dividend_payout]
        if dividend_payout and any(d > 0 for d in dividend_payout if d is not None):
            score += 1
            details.append("Company returns capital to shareholders through dividends.")
        else:
            details.append("Company doesn't pay dividends or data not available.")
    # Fallback to ratios if available
    elif dividend_data and "ratios" in dividend_data and "Dividend Payout Ratio" in dividend_data["ratios"]:
        dividend_payout = dividend_data["ratios"]["Dividend Payout Ratio"]
        if dividend_payout and any(d > 0 for d in dividend_payout if d is not None):
            score += 1
            details.append("Company returns capital to shareholders through dividends.")
        else:
            details.append("Company doesn't pay dividends or data not available.")
    else:
        details.append("Dividend payout data not available.")
    
    # 3. ROE consistency
    roe_data = get_roe(ticker)
    if roe_data:
        # Check for historical consistency
        if roe_data.get("is_consistently_high"):
            score += 2
            details.append("Consistently strong return on equity (>15%).")
        # If no historical consistency but current ROE is good
        elif roe_data.get("current_roe") is not None:
            current_roe = roe_data["current_roe"]
            roe_pct = current_roe * 100
            
            if current_roe > 0.15:
                score += 2
                details.append(f"Strong current return on equity ({roe_pct:.1f}%).")
            elif current_roe > 0.10:
                score += 1
                details.append(f"Decent current return on equity ({roe_pct:.1f}%).")
            else:
                details.append(f"Current return on equity ({roe_pct:.1f}%) below Ackman's typical threshold.")
        else:
            details.append("Return on equity data is inconsistent or below thresholds.")
    else:
        details.append("Return on equity data not available.")
    
    # Overall financial discipline score
    discipline_rating = "Excellent" if score >= 4 else "Good" if score >= 2 else "Average" if score >= 1 else "Poor"
    
    # Generate summary
    summary = f"Financial Discipline: {discipline_rating} ({score}/5 points)"
    if score >= 4:
        summary += ". Excellent capital allocation and shareholder returns."
    elif score >= 2:
        summary += ". Good balance sheet management with reasonable leverage."
    elif score >= 1:
        summary += ". Limited financial discipline metrics."
    else:
        summary += ". Poor financial management by Ackman's criteria."
    
    return {
        "score": score,
        "rating": discipline_rating,
        "details": details,
        "summary": summary
    }

def analyze_valuation(ticker: str, market_cap: Optional[float] = None) -> dict:
    """
    Simplified DCF based on latest Free Cash Flow and valuation metrics.
    """
    details = []
    fair_value_estimate = None
    score = 0  # Initialize score
    
    # Get market cap from tools if not provided
    if market_cap is None:
        market_cap_data = get_market_cap(ticker)
        if market_cap_data:
            market_cap = market_cap_data
    
    if not market_cap:
        details.append("Market cap data not available.")
        return {"rating": "Unknown", "details": details, "market_cap": market_cap, "score": 0, "summary": "Valuation: Unknown. Insufficient data."}
    
    # P/E ratio analysis
    pe_data = get_pe_ratio(ticker)
    pe_ratio = None
    pe_rating = "Unknown"  # Initialize PE-based rating
    
    if pe_data and isinstance(pe_data, dict) and "current_pe" in pe_data:
        pe_ratio = pe_data["current_pe"]
        
        if pe_ratio < 15:
            score += 2
            details.append(f"P/E ratio of {pe_ratio:.2f} appears undervalued.")
            pe_rating = "Potentially Undervalued"
        elif pe_ratio < 25:
            score += 1
            details.append(f"P/E ratio of {pe_ratio:.2f} appears fairly valued.")
            pe_rating = "Fairly Valued"
        else:
            details.append(f"P/E ratio of {pe_ratio:.2f} appears potentially overvalued.")
            pe_rating = "Potentially Overvalued"
    else:
        details.append("P/E ratio data not available.")
    
    # Free Cash Flow Yield using the new utility function
    fcf_data = get_free_cash_flow(ticker)
    fcf_yield = None
    fcf_based_fair_value = None
    
    if fcf_data and fcf_data.get("current_fcf") is not None:
        latest_fcf = fcf_data["current_fcf"]
        fcf_yield = fcf_data.get("fcf_yield")
        
        if fcf_yield is None and market_cap > 0 and latest_fcf != 0:
            fcf_yield = latest_fcf / market_cap
        
        if fcf_yield:
            # Always calculate fair value for reference even if below threshold
            fcf_based_fair_value = latest_fcf / 0.05  # 5% FCF yield as benchmark
            
            if fcf_yield > 0.05:  # 5% FCF yield benchmark
                score += 2
                details.append(f"FCF yield of {fcf_yield*100:.2f}% appears attractive.")
                fair_value_estimate = fcf_based_fair_value
            else:
                details.append(f"FCF yield of {fcf_yield*100:.2f}% is below Ackman's typical threshold.")
                # Still store the fair value even though it's below threshold
                fair_value_estimate = fcf_based_fair_value
        
        # Historical FCF trend information
        if fcf_data.get("is_positive"):
            score += 1
            details.append("Company has demonstrated consistent positive free cash flow generation.")
        else:
            details.append("Free cash flow has been inconsistent or negative in some periods.")
    else:
        # Fallback to direct data access
        fundamental_data = fetch_fundamental_data(ticker)
        if fundamental_data and "cash_flows" in fundamental_data and "Free Cash Flow" in fundamental_data["cash_flows"]:
            try:
                fcf_values = fundamental_data["cash_flows"]["Free Cash Flow"]
                if fcf_values and len(fcf_values) > 0:
                    latest_fcf = fcf_values[-1]
                    if market_cap > 0 and latest_fcf is not None and float(latest_fcf) != 0:
                        fcf_yield = float(latest_fcf) / market_cap
                        fcf_based_fair_value = latest_fcf / 0.05
                        fair_value_estimate = fcf_based_fair_value
                        
                        if fcf_yield > 0.05:
                            score += 2
                            details.append(f"FCF yield of {fcf_yield*100:.2f}% appears attractive.")
                        else:
                            details.append(f"FCF yield of {fcf_yield*100:.2f}% is below Ackman's typical threshold.")
                    else:
                        details.append("Invalid FCF or market cap values for yield calculation.")
                else:
                    details.append("No Free Cash Flow values available.")
            except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
                details.append("Free cash flow data not available.")
        else:
            details.append("Free cash flow data not available.")
    
    # Valuation conclusion
    rating = "Unknown"
    upside_text = ""
    
    # Calculate potential upside/downside if we have fair value estimate
    if fair_value_estimate:
        try:
            upside = (fair_value_estimate / market_cap - 1) * 100
            if upside > 20:
                rating = "Undervalued"
                score += 2
                details.append(f"Estimated {upside:.2f}% upside to fair value.")
                upside_text = f" with {upside:.1f}% upside potential"
            elif upside > 0:
                rating = "Slightly Undervalued"
                score += 1
                details.append(f"Estimated {upside:.2f}% upside to fair value.")
                upside_text = f" with {upside:.1f}% upside potential"
            elif upside > -20:
                rating = "Fairly Valued"
                details.append(f"Estimated {abs(upside):.2f}% downside to fair value.")
                upside_text = f" with {abs(upside):.1f}% downside risk"
            else:
                rating = "Overvalued"
                details.append(f"Estimated {abs(upside):.2f}% downside to fair value.")
                upside_text = f" with {abs(upside):.1f}% downside risk"
        except (TypeError, ValueError, ZeroDivisionError):
            # If calculation fails, use PE-based rating
            rating = pe_rating
            details.append("Unable to calculate precise valuation metrics, using P/E based assessment.")
    else:
        # If no fair value estimate, use PE ratio based rating
        rating = pe_rating
    
    # Generate summary
    value_rating = "Excellent" if score >= 4 else "Good" if score >= 2 else "Average" if score >= 1 else "Poor"
    summary = f"Valuation: {rating} ({score}/5 points){upside_text}"
    
    return {
        "rating": rating,
        "fair_value_estimate": fair_value_estimate,
        "market_cap": market_cap,
        "details": details,
        "score": score,
        "summary": summary
    }


class BillAckmanAgnoAgent():
    """Agno-based agent implementing Bill Ackman's activist investing principles."""
    
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
        prompt_template = """You are a Bill Ackman AI agent, making investment decisions using his principles:

                1. Seek high-quality businesses with durable competitive advantages (moats).
                2. Prioritize consistent free cash flow and growth potential.
                3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
                4. Look for temporarily undervalued companies with strong fundamentals.
                5. Concentrate investments in high-conviction ideas rather than diversifying broadly.

                Please analyze {ticker} using the following data:

                Business Quality Analysis:
                {business_quality}

                Financial Discipline Analysis:
                {financial_discipline}

                Valuation Analysis:
                {valuation}

                Based on this analysis and Bill Ackman's investment philosophy, would you recommend investing in {ticker}?
                Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
                Structure your reasoning to follow Ackman's activist investing approach.
                """
        return prompt_template.format(
            ticker=ticker,
            business_quality=json.dumps(analysis_summary["business_quality"], indent=2),
            financial_discipline=json.dumps(analysis_summary["financial_discipline"], indent=2),
            valuation=json.dumps(analysis_summary["valuation"], indent=2)
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Bill Ackman's principles.
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "bill_ackman_agno_agent"

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
                has_ratios = "ratios" in fundamental_data
                has_cash_flows = "cash_flows" in fundamental_data
                has_market_data = "market_data" in fundamental_data
                
                # Debug info about available data
                data_availability = {
                    "has_profit_loss": has_profit_loss,
                    "has_ratios": has_ratios,
                    "has_cash_flows": has_cash_flows,
                    "has_market_data": has_market_data
                }
                print(f"Data availability for {ticker}: {data_availability}")
                
                # Perform analysis using the direct API
                business_quality = analyze_business_quality(ticker)
                financial_discipline = analyze_financial_discipline(ticker)
                valuation = analyze_valuation(ticker)
                
                analysis = {
                    "business_quality": business_quality,
                    "financial_discipline": financial_discipline,
                    "valuation": valuation
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
    
    def _parse_response(self, response: str, ticker: str) -> BillAckmanSignal:
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
            
            return BillAckmanSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
            return BillAckmanSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )

    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Bill Ackman's investment criteria"""
        print(f"Starting Bill Ackman-based analysis for {ticker}")
        
        # Check if this is a banking entity
        try:
            is_bank = is_banking_entity(ticker)
            if is_bank:
                print(f"{ticker} is identified as a banking entity. Using specialized banking analysis.")
            else:
                print(f"{ticker} is not identified as a banking entity. Using standard analysis.")
        except Exception as e:
            print(f"Error checking if {ticker} is a banking entity: {e}")
            is_bank = False
        
        # Analyze the three key aspects of the investment
        business_quality = analyze_business_quality(ticker)
        financial_discipline = analyze_financial_discipline(ticker)
        valuation = analyze_valuation(ticker)
        
        # Calculate overall score (weighted average)
        business_weight = 0.4
        financial_weight = 0.3
        valuation_weight = 0.3
        overall_score = (
            business_quality["score"] * business_weight +
            financial_discipline["score"] * financial_weight +
            valuation["score"] * valuation_weight
        )
        
        # Determine investment recommendation
        if overall_score >= 8.0:
            recommendation = "Strong Buy"
        elif overall_score >= 7.0:
            recommendation = "Buy"
        elif overall_score >= 6.0:
            recommendation = "Hold"
        elif overall_score >= 4.0:
            recommendation = "Neutral"
        else:
            recommendation = "Avoid"
            
        # Add special banking-specific recommendation criteria if applicable
        if is_bank:
            # For banks, we want more conservative recommendations
            # Banks have higher systemic risks that aren't always reflected in the numbers
            
            # Check capital adequacy as a critical factor for banks
            car_data = get_bank_capital_adequacy(ticker)
            if car_data and car_data.get("current") is not None:
                car = car_data["current"]
                if car < 0.10:  # Less than 10% CAR is a red flag regardless of other metrics
                    recommendation = "Avoid - Inadequate Capital"
                    overall_score = min(overall_score, 4.0)  # Cap the score
            
            # Check if NIM (Net Interest Margin) is particularly low
            nim_data = get_net_interest_margin(ticker)
            if nim_data and nim_data.get("current") is not None:
                nim = nim_data["current"]
                if nim < 0.02:  # NIM below 2% is concerning for most banks
                    if recommendation in ["Buy", "Strong Buy"]:
                        recommendation = "Hold - Low NIM"
                        overall_score = min(overall_score, 6.0)  # Cap the score
        
        # Generate comprehensive report
        report = {
            "ticker": ticker,
            "is_bank": is_bank,
            "business_quality": business_quality,
            "financial_discipline": financial_discipline,
            "valuation": valuation,
            "overall_score": overall_score,
            "recommendation": recommendation,
            "summary": f"""
Bill Ackman Analysis for {ticker}{' (Banking Entity)' if is_bank else ''}:

Business Quality: {business_quality['score']:.1f}/10
{business_quality['summary']}

Financial Discipline: {financial_discipline['score']:.1f}/10
{financial_discipline['summary']}

Valuation: {valuation['score']:.1f}/10
{valuation['summary']}

Overall Score: {overall_score:.1f}/10
Recommendation: {recommendation}
"""
        }
        
        print(f"Completed Bill Ackman-based analysis for {ticker}")
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
        agent = BillAckmanAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.")