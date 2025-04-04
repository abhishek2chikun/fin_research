"""
Ben Graham Investing Agent using Agno Framework
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
from tools.fundamental import fetch_fundamental_data, get_market_cap, get_pe_ratio
from tools.fundamental import search_line_items

# Pydantic model for the output signal
class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def analyze_earnings_stability(ticker: str) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    # Get profit_loss data for EPS
    profit_loss = fetch_fundamental_data(ticker)
    if not profit_loss or "profit_loss" not in profit_loss:
        return {"score": score, "details": "Insufficient data for earnings stability analysis", "rating": "Poor", "summary": "Earnings Stability: Poor (0/4 points). Insufficient data."}

    # Extract EPS values
    eps_values = []
    if "EPS in Rs" in profit_loss["profit_loss"]:
        eps_values = profit_loss["profit_loss"]["EPS in Rs"]
        eps_values = [float(eps) if eps is not None else None for eps in eps_values]
    
    if len(eps_values) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details), "rating": "Poor", "summary": "Earnings Stability: Poor (0/4 points). Insufficient data."}

    # 1. Consistently positive EPS
    eps_values = [eps for eps in eps_values if eps is not None]
    positive_eps_years = sum(1 for e in eps_values if e > 0)
    total_eps_years = len(eps_values)
    
    if positive_eps_years == total_eps_years:
        score += 3
        details.append(f"EPS was positive in all {total_eps_years} available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append(f"EPS was positive in most periods ({positive_eps_years}/{total_eps_years}).")
    else:
        details.append(f"EPS was negative in multiple periods (positive in {positive_eps_years}/{total_eps_years}).")

    # 2. EPS growth from earliest to latest
    if eps_values[-1] > eps_values[0]:
        score += 1
        growth_pct = ((eps_values[-1] / eps_values[0]) - 1) * 100
        details.append(f"EPS grew by {growth_pct:.1f}% from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    # Determine rating based on score
    rating = "Excellent" if score >= 3 else "Good" if score >= 2 else "Average" if score >= 1 else "Poor"
    
    # Generate summary
    summary = f"Earnings Stability: {rating} ({score}/4 points)"
    if score >= 3:
        summary += ". Consistently positive earnings with growth."
    elif score >= 2:
        summary += ". Mostly positive earnings history."
    elif score >= 1:
        summary += ". Some positive earnings but inconsistent."
    else:
        summary += ". Poor earnings history."

    return {"score": score, "details": "; ".join(details), "rating": rating, "summary": summary}


def analyze_financial_strength(ticker: str) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    # Get balance sheet data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"score": score, "details": "No data for financial strength analysis", "rating": "Poor", "summary": "Financial Strength: Poor (0/5 points). No data available."}

    # Get latest balance sheet items
    balance_sheet = fundamental_data.get("balance_sheet", {})
    
    # Extract latest values
    dates = balance_sheet.get("date", [])
    if not dates:
        return {"score": score, "details": "No balance sheet dates available", "rating": "Poor", "summary": "Financial Strength: Poor (0/5 points). No balance sheet data."}
    
    # Get the latest index
    latest_idx = -1
    
    # Extract key metrics
    total_assets = None
    total_liabilities = None
    current_assets = None
    current_liabilities = None
    
    if "Total Assets" in balance_sheet and len(balance_sheet["Total Assets"]) > 0:
        try:
            total_assets = float(balance_sheet["Total Assets"][latest_idx])
        except (ValueError, TypeError):
            details.append("Total assets data not available or invalid.")
    else:
        details.append("Total assets data not available.")
        
    if "Total Liabilities" in balance_sheet and len(balance_sheet["Total Liabilities"]) > 0:
        try:
            total_liabilities = float(balance_sheet["Total Liabilities"][latest_idx])
        except (ValueError, TypeError):
            details.append("Total liabilities data not available or invalid.")
    else:
        details.append("Total liabilities data not available.")
        
    if "Current Assets" in balance_sheet and len(balance_sheet["Current Assets"]) > 0:
        try:
            current_assets = float(balance_sheet["Current Assets"][latest_idx])
        except (ValueError, TypeError):
            details.append("Current assets data not available or invalid.")
    else:
        details.append("Current assets data not available.")
        
    if "Current Liabilities" in balance_sheet and len(balance_sheet["Current Liabilities"]) > 0:
        try:
            current_liabilities = float(balance_sheet["Current Liabilities"][latest_idx])
        except (ValueError, TypeError):
            details.append("Current liabilities data not available or invalid.")
    else:
        details.append("Current liabilities data not available.")

    # 1. Current ratio
    if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current liabilities).")

    # 2. Debt vs. Assets
    if total_assets is not None and total_liabilities is not None and total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total assets or liabilities).")

    # 3. Dividend track record
    profit_loss = fundamental_data.get("profit_loss", {})
    dividend_payout = None
    
    if "Dividend Payout %" in profit_loss:
        dividend_payout = profit_loss["Dividend Payout %"]
    elif "Dividend %" in profit_loss:
        dividend_payout = profit_loss["Dividend %"]
    
    if dividend_payout:
        # Convert to proper format and handle possible errors
        div_periods = []
        for div in dividend_payout:
            try:
                if div is not None:
                    div_periods.append(float(div))
            except (ValueError, TypeError):
                continue
        
        if div_periods:
            # For Graham, we want to see consistent dividends
            div_paid_years = sum(1 for d in div_periods if d > 0)
            if div_paid_years > 0:
                if div_paid_years >= (len(div_periods) // 2 + 1):
                    score += 1
                    details.append(f"Company paid dividends in the majority of the reported years ({div_paid_years}/{len(div_periods)}).")
                else:
                    details.append(f"Company has some dividend payments ({div_paid_years}/{len(div_periods)}), but not most years.")
            else:
                details.append("Company did not pay dividends in these periods.")
        else:
            details.append("No valid dividend data available.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    # Determine rating based on score
    rating = "Excellent" if score >= 4 else "Good" if score >= 2 else "Average" if score >= 1 else "Poor"
    
    # Generate summary
    summary = f"Financial Strength: {rating} ({score}/5 points)"
    if score >= 4:
        summary += ". Excellent financial condition with strong balance sheet."
    elif score >= 2:
        summary += ". Reasonable financial condition."
    elif score >= 1:
        summary += ". Modest financial strength."
    else:
        summary += ". Weak financial condition."

    return {"score": score, "details": "; ".join(details), "rating": rating, "summary": summary}


def analyze_valuation_graham(ticker: str, market_cap: Optional[float] = None) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
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
        return {"score": 0, "details": "; ".join(details), "rating": "Unknown", "summary": "Valuation: Unknown. Insufficient data."}
    
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"score": 0, "details": "Insufficient data to perform valuation", "rating": "Unknown", "summary": "Valuation: Unknown. Insufficient data."}
    
    # Extract necessary data
    balance_sheet = fundamental_data.get("balance_sheet", {})
    profit_loss = fundamental_data.get("profit_loss", {})
    
    # Get latest indices
    latest_idx = -1
    
    # Extract key metrics for Graham valuation
    current_assets = None
    total_liabilities = None
    eps = None
    book_value_ps = None
    shares_outstanding = None
    
    # Current Assets
    if "Current Assets" in balance_sheet and len(balance_sheet["Current Assets"]) > 0:
        try:
            current_assets = float(balance_sheet["Current Assets"][latest_idx])
        except (ValueError, TypeError):
            details.append("Current assets data not available or invalid.")
    
    # Total Liabilities
    if "Total Liabilities" in balance_sheet and len(balance_sheet["Total Liabilities"]) > 0:
        try:
            total_liabilities = float(balance_sheet["Total Liabilities"][latest_idx])
        except (ValueError, TypeError):
            details.append("Total liabilities data not available or invalid.")
    
    # EPS
    if "EPS in Rs" in profit_loss and len(profit_loss["EPS in Rs"]) > 0:
        try:
            eps = float(profit_loss["EPS in Rs"][latest_idx])
        except (ValueError, TypeError):
            details.append("EPS data not available or invalid.")
    
    # Book Value per Share
    if "Book Value" in fundamental_data.get("market_data", {}):
        try:
            book_value_ps = float(fundamental_data["market_data"]["Book Value"])
        except (ValueError, TypeError):
            details.append("Book value per share data not available or invalid.")
    
    # Outstanding Shares
    if "No. of Shares" in fundamental_data.get("market_data", {}):
        try:
            shares_outstanding = float(fundamental_data["market_data"]["No. of Shares"])
        except (ValueError, TypeError):
            # Try to calculate from market cap and current price
            if "Current Price" in fundamental_data.get("market_data", {}):
                try:
                    current_price = float(fundamental_data["market_data"]["Current Price"])
                    if current_price > 0:
                        shares_outstanding = market_cap / current_price
                except (ValueError, TypeError, ZeroDivisionError):
                    details.append("Outstanding shares data not available or invalid.")
            else:
                details.append("Outstanding shares data not available.")
    
    # 1. Net-Net Check
    if current_assets is not None and total_liabilities is not None and shares_outstanding is not None:
        net_current_asset_value = current_assets - total_liabilities
        
        if net_current_asset_value > 0 and shares_outstanding > 0:
            net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
            price_per_share = market_cap / shares_outstanding
            
            details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
            details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
            details.append(f"Price Per Share = {price_per_share:,.2f}")
            
            if net_current_asset_value > market_cap:
                score += 4  # Very strong Graham signal
                details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
            else:
                # For partial net-net discount
                if net_current_asset_value_per_share >= (price_per_share * 0.67):
                    score += 2
                    details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
        else:
            details.append("NCAV calculation yielded invalid results; can't perform net-net analysis.")
    else:
        details.append("Insufficient data for net-net calculation.")
    
    # 2. Graham Number
    graham_number = None
    if eps is not None and book_value_ps is not None and eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")
    
    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding is not None and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price calculation invalid; can't compute margin of safety.")
    
    # Calculate PE ratio for additional context
    pe_data = get_pe_ratio(ticker)
    if pe_data and "current_pe" in pe_data:
        pe_ratio = pe_data["current_pe"]
        details.append(f"Current P/E ratio = {pe_ratio:.2f}")
        
        # Graham typically preferred P/E ratios below 15
        if pe_ratio < 15:
            details.append("P/E ratio is below Graham's preferred maximum of 15.")
        else:
            details.append("P/E ratio exceeds Graham's preferred maximum of 15.")
    
    # Determine rating based on score
    if score >= 5:
        rating = "Undervalued"
    elif score >= 3:
        rating = "Fairly Valued"
    elif score >= 1:
        rating = "Slightly Overvalued"
    else:
        rating = "Overvalued" if graham_number else "Unknown"
    
    # Generate summary
    summary = f"Valuation: {rating} ({score}/7 points)"
    if score >= 5:
        summary += ". Strong Graham value with significant margin of safety."
    elif score >= 3:
        summary += ". Moderate value with some margin of safety."
    elif score >= 1:
        summary += ". Limited value metrics."
    else:
        summary += ". No significant Graham value indicators found."
    
    return {
        "score": score, 
        "details": "; ".join(details), 
        "rating": rating, 
        "summary": summary,
        "graham_number": graham_number
    }


class BenGrahamAgnoAgent():
    """Agno-based agent implementing Benjamin Graham's value investing principles."""
    
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
        prompt_template = """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
            
            Please analyze {ticker} using the following data:

            Earnings Stability Analysis:
            {earnings_stability}

            Financial Strength Analysis:
            {financial_strength}

            Valuation Analysis:
            {valuation}

            Based on this analysis and Benjamin Graham's investment philosophy, would you recommend investing in {ticker}?
            Provide a clear signal (bullish, bearish, or neutral) with a confidence score (0.0 to 1.0).
            Structure your reasoning to follow Graham's value investing approach.
            """
        return prompt_template.format(
            ticker=ticker,
            earnings_stability=json.dumps(analysis_summary["earnings_stability"], indent=2),
            financial_strength=json.dumps(analysis_summary["financial_strength"], indent=2),
            valuation=json.dumps(analysis_summary["valuation"], indent=2)
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using Benjamin Graham's principles.
        """
        if not self.agent:
            raise RuntimeError("Agno agent not initialized.")

        agent_name = "ben_graham_agno_agent"

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
                has_market_data = "market_data" in fundamental_data
                
                # Debug info about available data
                data_availability = {
                    "has_profit_loss": has_profit_loss,
                    "has_balance_sheet": has_balance_sheet,
                    "has_market_data": has_market_data
                }
                print(f"Data availability for {ticker}: {data_availability}")
                
                # Perform analysis using the direct API
                earnings_stability = analyze_earnings_stability(ticker)
                financial_strength = analyze_financial_strength(ticker)
                valuation = analyze_valuation_graham(ticker)
                
                analysis = {
                    "earnings_stability": earnings_stability,
                    "financial_strength": financial_strength,
                    "valuation": valuation
                }
                
                # Calculate overall score and max score
                total_score = (
                    earnings_stability.get("score", 0) + 
                    financial_strength.get("score", 0) + 
                    valuation.get("score", 0)
                )
                max_score = 16  # 4 + 5 + 7
                
                # Map total_score to signal
                if total_score >= 0.7 * max_score:
                    signal = "bullish"
                elif total_score <= 0.3 * max_score:
                    signal = "bearish"
                else:
                    signal = "neutral"
                
                analysis["signal"] = signal
                analysis["score"] = total_score
                analysis["max_score"] = max_score
                
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
    
    def _parse_response(self, response: str, ticker: str) -> BenGrahamSignal:
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
            
            return BenGrahamSignal(
                signal=signal,
                confidence=min(1.0, max(0.0, confidence)),  # Ensure between 0 and 1
                reasoning=reasoning
            )
        except Exception as e:
            # Default to neutral if parsing fails
            return BenGrahamSignal(
                signal="neutral",
                confidence=0.5,
                reasoning=f"Error parsing response for {ticker}: {str(e)}"
            )

    def analyze(self, ticker: str) -> Dict:
        """Analyze a stock based on Benjamin Graham's investment criteria"""
        print(f"Starting Benjamin Graham-based analysis for {ticker}")
        
        # Analyze the three key aspects of the investment
        earnings_stability = analyze_earnings_stability(ticker)
        financial_strength = analyze_financial_strength(ticker)
        valuation = analyze_valuation_graham(ticker)
        
        # Calculate overall score
        total_score = (
            earnings_stability.get("score", 0) + 
            financial_strength.get("score", 0) + 
            valuation.get("score", 0)
        )
        max_score = 16  # 4 + 5 + 7
        
        # Determine investment recommendation
        if total_score >= 11:  # ~70% of max score
            recommendation = "Strong Buy"
        elif total_score >= 8:  # ~50% of max score
            recommendation = "Buy"
        elif total_score >= 5:  # ~30% of max score
            recommendation = "Hold"
        else:
            recommendation = "Avoid"
        
        # Generate comprehensive report
        report = {
            "ticker": ticker,
            "earnings_stability": earnings_stability,
            "financial_strength": financial_strength,
            "valuation": valuation,
            "overall_score": total_score,
            "max_score": max_score,
            "recommendation": recommendation,
            "summary": f"""
Benjamin Graham Analysis for {ticker}:

{earnings_stability['summary']}

{financial_strength['summary']}

{valuation['summary']}

Overall Score: {total_score}/{max_score}
Recommendation: {recommendation}
"""
        }
        
        print(f"Completed Benjamin Graham-based analysis for {ticker}")
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
        agent = BenGrahamAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure FundamentalData is properly set up.") 