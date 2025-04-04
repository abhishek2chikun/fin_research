import json
import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to fundamental data
FUNDAMENTAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Historical_data/fundamental")

def load_ticker_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Load the raw JSON data for a given ticker symbol.
    
    Args:
        ticker: The ticker symbol to load data for
        
    Returns:
        The JSON data as a dictionary or None if the file doesn't exist
    """
    try:
        file_path = os.path.join(FUNDAMENTAL_DATA_PATH, f"{ticker}.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"No fundamental data found for {ticker}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON data for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None

# Alias for compatibility with legacy code
fetch_fundamental_data = load_ticker_data

def list_available_tickers() -> List[str]:
    """
    List all available ticker symbols with fundamental data.
    
    Returns:
        List of available ticker symbols
    """
    try:
        files = os.listdir(FUNDAMENTAL_DATA_PATH)
        return [f[:-5] for f in files if f.endswith('.json')]
    except Exception as e:
        logger.error(f"Error listing available tickers: {str(e)}")
        return []

def safe_float_convert(value: Any) -> Optional[float]:
    """
    Safely convert a value to float, handling various formats.
    
    Args:
        value: The value to convert
        
    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove commas, percentage signs, etc.
        clean_value = re.sub(r'[,₹%]', '', value.strip())
        try:
            return float(clean_value)
        except ValueError:
            return None
    
    return None

def get_section(ticker: str, section_name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific section from the ticker data.
    
    Args:
        ticker: The ticker symbol
        section_name: Name of the section to retrieve
        
    Returns:
        Section data as a dictionary or None if not found
    """
    data = load_ticker_data(ticker)
    if not data or section_name not in data:
        return None
    return data[section_name]

def get_quarterly_results(ticker: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get quarterly results data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "quarterly_results")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Handle various formats of the same key
        possible_keys = [key, f"{key}\u00a0+", f"{key} %", f"{key}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return entire quarterly results
    return results

def get_profit_loss(ticker: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get profit and loss data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "profit_loss")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Handle various formats of the same key
        possible_keys = [key, f"{key}\u00a0+", f"{key} %", f"{key}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return entire profit loss data
    return results

def get_balance_sheet(ticker: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get balance sheet data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "balance_sheet")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Handle various formats of the same key
        possible_keys = [key, f"{key}\u00a0+", f"{key} %", f"{key}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return entire balance sheet data
    return results

def get_cash_flows(ticker: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get cash flow data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "cash_flows")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Handle various formats of the same key
        possible_keys = [key, f"{key}\u00a0+", f"{key} %", f"{key}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return entire cash flow data
    return results

def get_ratios(ticker: str, key: Optional[str] = None) -> Dict[str, Any]:
    """
    Get financial ratios data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "ratios")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Handle various formats of the same key
        possible_keys = [key, f"{key}\u00a0+", f"{key} %", f"{key}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return all ratios
    return results

def get_compounded_growth(ticker: str, growth_type: str) -> Dict[str, Any]:
    """
    Get compounded growth data for a ticker.
    
    Args:
        ticker: The ticker symbol
        growth_type: Type of growth data to retrieve ('sales', 'profit', 'stock_price')
        
    Returns:
        Dictionary with the data or empty dict if not found
    """
    section_map = {
        "sales": "compounded_sales_growth",
        "profit": "compounded_profit_growth",
        "stock_price": "stock_price_cagr"
    }
    
    if growth_type not in section_map:
        logger.error(f"Unknown growth type: {growth_type}")
        return {}
    
    section = section_map[growth_type]
    results = get_section(ticker, section)
    
    if not results:
        return {}
    
    return {
        "dates": results.get("date", []),
        "values": results.get(list(results.keys())[-1], [])  # Last key should contain the values
    }

def get_return_on_equity(ticker: str) -> Dict[str, Any]:
    """
    Get Return on Equity data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with the ROE data or empty dict if not found
    """
    # First try to get from the dedicated ROE section
    results = get_section(ticker, "return_on_equity")
    if results:
        return {
            "dates": results.get("date", []),
            "values": results.get("Return On Equity", [])
        }
    
    # Try to get from ratios section
    ratios = get_ratios(ticker, "ROE")
    if ratios and "values" in ratios and ratios["values"]:
        return ratios
    
    # Try to get from market data
    market_data = get_market_data(ticker)
    if market_data and "ROE" in market_data:
        return {
            "current": safe_float_convert(market_data["ROE"])
        }
    
    return {}

def get_shareholding_pattern(ticker: str, holder_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get shareholding pattern data for a ticker.
    
    Args:
        ticker: The ticker symbol
        holder_type: Optional holder type (e.g., 'Promoters', 'FIIs')
    
    Returns:
        Dictionary with the data or empty dict if not found
    """
    results = get_section(ticker, "shareholding_pattern")
    if not results:
        return {}
    
    # If a specific holder type is requested
    if holder_type:
        # Handle various formats of the same key
        possible_keys = [holder_type, f"{holder_type}\u00a0+", f"{holder_type} %", f"{holder_type}%"]
        
        for possible_key in possible_keys:
            if possible_key in results:
                return {
                    "dates": results.get("date", []),
                    "values": results.get(possible_key, [])
                }
        
        return {}
    
    # Return all shareholding pattern data
    return results

def get_market_data(ticker: str, key: Optional[str] = None) -> Union[Dict[str, Any], Optional[float]]:
    """
    Get market data for a ticker.
    
    Args:
        ticker: The ticker symbol
        key: Optional specific data key to retrieve
        
    Returns:
        Dictionary with all data, a specific value, or None if not found
    """
    results = get_section(ticker, "market_data")
    if not results:
        return {}
    
    # If a specific key is requested
    if key:
        # Try direct key access
        if key in results:
            return safe_float_convert(results[key])
        
        # Try with different formats
        formatted_key = key.replace('_', ' ')
        if formatted_key in results:
            return safe_float_convert(results[formatted_key])
        
        return None
    
    # Return all market data
    return results

def get_remarks(ticker: str, remark_type: Optional[str] = None) -> Union[Dict[str, Any], List[str]]:
    """
    Get remarks data for a ticker.
    
    Args:
        ticker: The ticker symbol
        remark_type: Optional remark type ('pros' or 'cons')
        
    Returns:
        Dictionary with all remarks or list of specific remarks
    """
    results = get_section(ticker, "remarks")
    if not results:
        return {}
    
    # If a specific remark type is requested
    if remark_type and remark_type in results:
        return results[remark_type]
    
    # Return all remarks
    return results

def is_banking_entity(ticker: str) -> bool:
    """
    Determine if a company is a banking or financial institution.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        True if the company is a banking entity, False otherwise
    """
    data = load_ticker_data(ticker)
    if not data:
        return False
    
    # Check for typical banking indicators in the data structure
    
    # Check if 'Deposits' exists in balance sheet
    if "balance_sheet" in data:
        if "Deposits" in data["balance_sheet"]:
            return True
    
    # Check for banking-specific metrics
    banking_indicators = [
        "Net Interest Margin",
        "Gross NPA %", 
        "Net NPA %",
        "Financing Margin %",
        "Capital Adequacy Ratio"
    ]
    
    # Check quarterly results
    if "quarterly_results" in data:
        for indicator in banking_indicators:
            if indicator in data["quarterly_results"]:
                return True
            # Check with non-breaking space
            if f"{indicator}\u00a0+" in data["quarterly_results"]:
                return True
    
    # Additional checks for banking-specific terms
    if "profit_loss" in data:
        if "Interest" in data["profit_loss"] or "Financing Profit" in data["profit_loss"]:
            return True
    
    return False

def get_net_interest_margin(ticker: str) -> Dict[str, Any]:
    """
    Get Net Interest Margin (NIM) data for a banking entity.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with NIM data or empty dict if not found or not a bank
    """
    if not is_banking_entity(ticker):
        return {}
    
    # Try to find NIM in quarterly results
    quarterly = get_quarterly_results(ticker, "Financing Margin")
    if quarterly and "values" in quarterly and quarterly["values"]:
        values = [safe_float_convert(v) for v in quarterly["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": quarterly.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
            }
    
    # Try to find NIM in profit_loss
    profit_loss = get_profit_loss(ticker, "Financing Margin")
    if profit_loss and "values" in profit_loss and profit_loss["values"]:
        values = [safe_float_convert(v) for v in profit_loss["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": profit_loss.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
            }
    
    # Try to calculate NIM from interest and revenue
    interest_data = get_profit_loss(ticker, "Interest")
    revenue_data = get_profit_loss(ticker, "Revenue")
    
    if (interest_data and "values" in interest_data and interest_data["values"] and
        revenue_data and "values" in revenue_data and revenue_data["values"]):
        
        interest_values = [safe_float_convert(v) for v in interest_data["values"]]
        revenue_values = [safe_float_convert(v) for v in revenue_data["values"]]
        
        # Calculate NIM as Interest/Revenue
        if len(interest_values) == len(revenue_values):
            nim_values = []
            for i, r in zip(interest_values, revenue_values):
                if i is not None and r is not None and r != 0:
                    nim_values.append((i / r) * 100)  # Convert to percentage
                else:
                    nim_values.append(None)
            
            valid_values = [v for v in nim_values if v is not None]
            if valid_values:
                return {
                    "dates": revenue_data.get("dates", []),
                    "values": nim_values,
                    "current": valid_values[-1] if valid_values else None,
                    "average": sum(valid_values) / len(valid_values) if valid_values else None,
                    "trend": "improving" if valid_values[-1] > valid_values[0] else "declining" if valid_values[-1] < valid_values[0] else "stable"
                }
    
    return {}

def get_bank_capital_adequacy(ticker: str) -> Dict[str, Any]:
    """
    Get Capital Adequacy Ratio (CAR) data for a banking entity.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with CAR data or empty dict if not found or not a bank
    """
    if not is_banking_entity(ticker):
        return {}
    
    # Try to find CAR in ratios
    ratios = get_ratios(ticker, "Capital Adequacy Ratio")
    if ratios and "values" in ratios and ratios["values"]:
        values = [safe_float_convert(v) for v in ratios["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": ratios.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
            }
    
    # Alternative names for CAR
    for alt_name in ["CAR %", "CRAR %", "Total Capital Ratio"]:
        ratios = get_ratios(ticker, alt_name)
        if ratios and "values" in ratios and ratios["values"]:
            values = [safe_float_convert(v) for v in ratios["values"] if safe_float_convert(v) is not None]
            if values:
                return {
                    "dates": ratios.get("dates", []),
                    "values": values,
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
                }
    
    # Look for CAR in quarterly results
    quarterly = get_quarterly_results(ticker, "Capital Adequacy Ratio")
    if quarterly and "values" in quarterly and quarterly["values"]:
        values = [safe_float_convert(v) for v in quarterly["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": quarterly.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
            }
    
    return {}

def get_cost_to_income_ratio(ticker: str) -> Dict[str, Any]:
    """
    Get Cost to Income Ratio data for a banking entity.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with Cost to Income Ratio data or empty dict if not found or not a bank
    """
    if not is_banking_entity(ticker):
        return {}
    
    # Try to find the ratio directly
    for ratio_name in ["Cost to Income Ratio", "Cost/Income", "Cost to Income"]:
        ratios = get_ratios(ticker, ratio_name)
        if ratios and "values" in ratios and ratios["values"]:
            values = [safe_float_convert(v) for v in ratios["values"] if safe_float_convert(v) is not None]
            if values:
                return {
                    "dates": ratios.get("dates", []),
                    "values": values,
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "trend": "improving" if values[-1] < values[0] else "declining" if values[-1] > values[0] else "stable"
                }
    
    # Try to calculate from expenses and revenue
    expenses_data = get_profit_loss(ticker, "Expenses")
    revenue_data = get_profit_loss(ticker, "Revenue")
    
    if (expenses_data and "values" in expenses_data and expenses_data["values"] and
        revenue_data and "values" in revenue_data and revenue_data["values"]):
        
        expense_values = [safe_float_convert(v) for v in expenses_data["values"]]
        revenue_values = [safe_float_convert(v) for v in revenue_data["values"]]
        
        # Calculate ratio as Expenses/Revenue
        if len(expense_values) == len(revenue_values):
            ratio_values = []
            for e, r in zip(expense_values, revenue_values):
                if e is not None and r is not None and r != 0:
                    ratio_values.append((e / r) * 100)  # Convert to percentage
                else:
                    ratio_values.append(None)
            
            valid_values = [v for v in ratio_values if v is not None]
            if valid_values:
                return {
                    "dates": revenue_data.get("dates", []),
                    "values": ratio_values,
                    "current": valid_values[-1] if valid_values else None,
                    # Lower cost to income is better, so if it's decreasing, it's improving
                    "trend": "improving" if valid_values[-1] < valid_values[0] else "declining" if valid_values[-1] > valid_values[0] else "stable"
                }
    
    return {}

def get_price_to_book(ticker: str) -> Dict[str, Any]:
    """
    Get Price to Book (P/B) Ratio data.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with P/B data or empty dict if not found
    """
    # First try to get from market data
    market_data = get_market_data(ticker)
    if market_data:
        # Try direct access
        if "Price/Book" in market_data:
            pb_value = safe_float_convert(market_data["Price/Book"])
            if pb_value is not None:
                return {"current": pb_value}
        
        # Alternative field names
        for field in ["Price to Book", "P/B", "P/BV"]:
            if field in market_data:
                pb_value = safe_float_convert(market_data[field])
                if pb_value is not None:
                    return {"current": pb_value}
        
        # Try calculating from available data
        if "Current Price" in market_data and "Book Value" in market_data:
            price = safe_float_convert(market_data["Current Price"])
            book_value = safe_float_convert(market_data["Book Value"])
            
            if price is not None and book_value is not None and book_value != 0:
                pb_value = price / book_value
                return {"current": pb_value}
    
    # Try to get from ratios over time
    ratios = get_ratios(ticker, "Price to Book")
    if ratios and "values" in ratios and ratios["values"]:
        values = [safe_float_convert(v) for v in ratios["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": ratios.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
            }
    
    return {}

def get_ev_to_ebitda(ticker: str) -> Dict[str, Any]:
    """
    Get EV/EBITDA Ratio data for a non-banking entity.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with EV/EBITDA data or empty dict if a bank or not found
    """
    if is_banking_entity(ticker):
        return {}  # EV/EBITDA not applicable to banks
    
    # First try to get from market data
    market_data = get_market_data(ticker)
    if market_data:
        for field in ["EV/EBITDA", "Enterprise Value/EBITDA", "EV to EBITDA"]:
            if field in market_data:
                ev_ebitda_value = safe_float_convert(market_data[field])
                if ev_ebitda_value is not None:
                    return {"current": ev_ebitda_value}
    
    # Try to get from ratios over time
    ratios = get_ratios(ticker, "EV/EBITDA")
    if ratios and "values" in ratios and ratios["values"]:
        values = [safe_float_convert(v) for v in ratios["values"] if safe_float_convert(v) is not None]
        if values:
            return {
                "dates": ratios.get("dates", []),
                "values": values,
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else None,
                "trend": "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
            }
    
    return {}

def get_market_cap(ticker: str) -> Optional[float]:
    """
    Get the market capitalization for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Market cap as float or None if not found
    """
    market_data = get_market_data(ticker)
    if not market_data:
        return None
    
    # Try different field names
    for field in ["Market Cap", "Market_Cap", "MarketCap"]:
        if field in market_data:
            return safe_float_convert(market_data[field])
    
    return None

def search_line_items(ticker: str) -> List[Dict[str, Any]]:
    """
    Search and return financial line items for analysis.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        List of financial line items with calculated key ratios
    """
    # Get balance sheet and profit/loss data
    balance_sheet = get_balance_sheet(ticker)
    profit_loss = get_profit_loss(ticker)
    
    if not balance_sheet or not profit_loss:
        return []
    
    # Extract dates, assets, liabilities
    dates = balance_sheet.get("date", [])
    if not dates:
        return []
    
    line_items = []
    
    for i, date in enumerate(dates):
        if i >= len(dates):
            break
            
        item = {"date": date}
        
        # Balance sheet items
        if "Total Assets" in balance_sheet:
            if i < len(balance_sheet["Total Assets"]):
                item["total_assets"] = safe_float_convert(balance_sheet["Total Assets"][i])
                
        if "Current Assets" in balance_sheet:
            if i < len(balance_sheet["Current Assets"]):
                item["current_assets"] = safe_float_convert(balance_sheet["Current Assets"][i])
                
        if "Total Liabilities" in balance_sheet:
            if i < len(balance_sheet["Total Liabilities"]):
                item["total_liabilities"] = safe_float_convert(balance_sheet["Total Liabilities"][i])
                
        if "Current Liabilities" in balance_sheet:
            if i < len(balance_sheet["Current Liabilities"]):
                item["current_liabilities"] = safe_float_convert(balance_sheet["Current Liabilities"][i])
        
        # Add profit/loss items if we can match the date
        if date in profit_loss.get("date", []):
            pl_idx = profit_loss["date"].index(date)
            
            if "EPS in Rs" in profit_loss and pl_idx < len(profit_loss["EPS in Rs"]):
                item["earnings_per_share"] = safe_float_convert(profit_loss["EPS in Rs"][pl_idx])
                
            if "Net Profit" in profit_loss and pl_idx < len(profit_loss["Net Profit"]):
                item["net_profit"] = safe_float_convert(profit_loss["Net Profit"][pl_idx])
                
            if "Dividend Payout %" in profit_loss and pl_idx < len(profit_loss["Dividend Payout %"]):
                item["dividends_and_other_cash_distributions"] = safe_float_convert(profit_loss["Dividend Payout %"][pl_idx])
        
        # Calculate additional metrics
        if "total_assets" in item and "total_liabilities" in item:
            equity = item["total_assets"] - item["total_liabilities"]
            item["total_equity"] = equity
            
            # If we have EPS, we can estimate outstanding shares and book value per share
            if "earnings_per_share" in item and item["earnings_per_share"] > 0 and "net_profit" in item:
                # Estimate outstanding shares from net profit and EPS
                eps = item["earnings_per_share"]
                if eps > 0:
                    shares = item["net_profit"] / eps
                    item["outstanding_shares"] = shares
                    
                    # Book value per share
                    if shares > 0:
                        item["book_value_per_share"] = equity / shares
        
        line_items.append(item)
    
    return line_items

def get_financial_metrics(ticker: str) -> List[Dict[str, Any]]:
    """
    Get key financial metrics for analysis.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        List of financial metrics
    """
    # For now, we'll return the same data as search_line_items
    # In a full implementation, this would have additional metrics
    return search_line_items(ticker)

def get_revenue_growth(ticker: str) -> Dict[str, Any]:
    """
    Get revenue growth data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with revenue growth data
    """
    # Try to get from compounded sales growth
    compounded_growth = get_section(ticker, "compounded_sales_growth")
    if compounded_growth and "date" in compounded_growth and "Compounded Sales Growth" in compounded_growth:
        dates = compounded_growth["date"]
        growth_values = compounded_growth["Compounded Sales Growth"]
        
        if dates and growth_values and len(dates) > 0 and len(growth_values) > 0:
            # Convert to float and handle percentage values
            growth_values = [safe_float_convert(v) for v in growth_values]
            growth_values = [v/100 if v and v > 1 else v for v in growth_values]
            
            return {
                "dates": dates,
                "growth_values": growth_values,
                "growth_rate": growth_values[0] if growth_values else None,
                "trend": "increasing" if len(growth_values) > 1 and growth_values[0] > growth_values[-1] else "decreasing" if len(growth_values) > 1 and growth_values[0] < growth_values[-1] else "stable"
            }
    
    # Try to calculate from profit_loss revenue data
    profit_loss = get_profit_loss(ticker)
    if profit_loss and "date" in profit_loss and "Revenue" in profit_loss:
        dates = profit_loss["date"]
        revenue = profit_loss["Revenue"]
        
        if dates and revenue and len(dates) > 1 and len(revenue) > 1:
            # Convert to float
            revenue_values = [safe_float_convert(v) for v in revenue]
            
            if len(revenue_values) > 1 and revenue_values[0] and revenue_values[-1]:
                # Calculate CAGR if we have enough years
                years = len(revenue_values)
                if years >= 2:
                    start_revenue = revenue_values[0]
                    end_revenue = revenue_values[-1]
                    if start_revenue > 0:
                        cagr = (end_revenue / start_revenue) ** (1 / years) - 1
                        return {
                            "dates": dates,
                            "revenue_values": revenue_values,
                            "growth_rate": cagr,
                            "start_revenue": start_revenue,
                            "end_revenue": end_revenue,
                            "years": years
                        }
    
    return {"growth_rate": None}

def get_roe(ticker: str) -> Dict[str, Any]:
    """
    Get Return on Equity (ROE) data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with ROE data
    """
    # Try to get from ratios
    ratios = get_ratios(ticker, "ROE %")
    
    if ratios and "dates" in ratios and "values" in ratios:
        values = ratios["values"]
        
        # Convert to float and handle percentage values
        roe_values = [safe_float_convert(v) for v in values]
        roe_values = [v/100 if v and v > 1 else v for v in roe_values]
        
        if roe_values and len(roe_values) > 0:
            current_roe = roe_values[-1]
            is_consistently_high = all(v and v >= 0.15 for v in roe_values if v is not None)
            
            return {
                "dates": ratios["dates"],
                "values": roe_values,
                "current_roe": current_roe,
                "is_consistently_high": is_consistently_high,
                "average": sum(v for v in roe_values if v is not None) / len([v for v in roe_values if v is not None]) if roe_values else None
            }
    
    # Try to get from market_data
    market_data = get_market_data(ticker)
    if market_data and "ROE" in market_data:
        roe = safe_float_convert(market_data["ROE"])
        if roe is not None:
            # Convert to decimal if it's a percentage
            if roe > 1:
                roe = roe / 100
            
            return {
                "current_roe": roe,
                "is_consistently_high": roe >= 0.15
            }
    
    return {}

def get_pe_ratio(ticker: str) -> Dict[str, Any]:
    """
    Get Price to Earnings (P/E) ratio data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with P/E ratio data
    """
    # Try to get from market data first
    market_data = get_market_data(ticker)
    if market_data:
        # Look for different field names for PE ratio
        for field in ["Stock P/E", "P/E", "PE Ratio", "Price/Earnings"]:
            if field in market_data:
                pe = safe_float_convert(market_data[field])
                if pe is not None:
                    return {
                        "current_pe": pe,
                        "is_reasonable": pe < 25,  # Using 25 as a conservative threshold
                        "source": "market_data"
                    }
    
    # Try to calculate from price and EPS
    try:
        current_price = None
        if market_data and "Current Price" in market_data:
            current_price = safe_float_convert(market_data["Current Price"])
        
        if current_price:
            # Get latest EPS
            profit_loss = get_profit_loss(ticker)
            if profit_loss and "EPS in Rs" in profit_loss:
                eps_values = profit_loss["EPS in Rs"]
                if eps_values and len(eps_values) > 0:
                    latest_eps = safe_float_convert(eps_values[-1])
                    if latest_eps and latest_eps > 0:
                        pe_ratio = current_price / latest_eps
                        return {
                            "current_pe": pe_ratio,
                            "is_reasonable": pe_ratio < 25,
                            "current_price": current_price,
                            "latest_eps": latest_eps,
                            "source": "calculated"
                        }
    except (TypeError, ZeroDivisionError):
        pass
    
    return {}

def get_operating_margin(ticker: str) -> Dict[str, Any]:
    """
    Get Operating Margin data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with Operating Margin data
    """
    # Get profit_loss data
    profit_loss = get_profit_loss(ticker)
    
    if profit_loss and "date" in profit_loss:
        dates = profit_loss["date"]
        
        # Look for operating margin in different formats
        for field in ["OPM %", "Operating Profit Margin %", "Operating Margin %"]:
            if field in profit_loss:
                margin_values = profit_loss[field]
                
                if margin_values and len(margin_values) > 0:
                    # Convert to float and handle percentage values
                    margins = [safe_float_convert(v) for v in margin_values]
                    margins = [v/100 if v and v > 1 else v for v in margins]
                    
                    if margins:
                        current_margin = margins[-1] if margins else None
                        is_consistently_high = all(v and v >= 0.15 for v in margins if v is not None)
                        
                        return {
                            "dates": dates,
                            "historical_margins": margins,
                            "current_operating_margin": current_margin,
                            "is_consistently_high": is_consistently_high,
                            "average": sum(v for v in margins if v is not None) / len([v for v in margins if v is not None]) if margins else None
                        }
        
        # If we can't find a direct operating margin field, try to calculate it
        if "Operating Profit" in profit_loss and "Revenue" in profit_loss:
            op_profit = profit_loss["Operating Profit"]
            revenue = profit_loss["Revenue"]
            
            if op_profit and revenue and len(op_profit) == len(revenue):
                margins = []
                for i in range(len(op_profit)):
                    op = safe_float_convert(op_profit[i])
                    rev = safe_float_convert(revenue[i])
                    
                    if op is not None and rev is not None and rev > 0:
                        margins.append(op / rev)
                    else:
                        margins.append(None)
                
                if margins:
                    current_margin = margins[-1] if margins else None
                    is_consistently_high = all(v and v >= 0.15 for v in margins if v is not None)
                    
                    return {
                        "dates": dates,
                        "historical_margins": margins,
                        "current_operating_margin": current_margin,
                        "is_consistently_high": is_consistently_high,
                        "average": sum(v for v in margins if v is not None) / len([v for v in margins if v is not None]) if margins else None,
                        "source": "calculated"
                    }
        
        # For banking entities, try using the financing margin
        if is_banking_entity(ticker) and "Financing Margin %" in profit_loss:
            margin_values = profit_loss["Financing Margin %"]
            
            if margin_values and len(margin_values) > 0:
                # Convert to float and handle percentage values
                margins = [safe_float_convert(v) for v in margin_values]
                margins = [v/100 if v and v > 1 else v for v in margins]
                
                if margins:
                    current_margin = margins[-1] if margins else None
                    is_consistently_high = all(v and v >= 0.15 for v in margins if v is not None)
                    
                    return {
                        "dates": dates,
                        "historical_margins": margins,
                        "current_operating_margin": current_margin,
                        "is_consistently_high": is_consistently_high,
                        "average": sum(v for v in margins if v is not None) / len([v for v in margins if v is not None]) if margins else None,
                        "source": "financing_margin"
                    }
    
    return {}

def get_debt_to_equity(ticker: str) -> Dict[str, Any]:
    """
    Get Debt to Equity ratio data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with Debt to Equity data
    """
    # Get balance sheet data
    balance_sheet = get_balance_sheet(ticker)
    
    if balance_sheet and "date" in balance_sheet:
        dates = balance_sheet["date"]
        
        # For a standard company, use Total Liabilities and (Equity + Reserves)
        if "Total Liabilities" in balance_sheet and "Equity Capital" in balance_sheet and "Reserves" in balance_sheet:
            liabilities = balance_sheet["Total Liabilities"]
            equity = balance_sheet["Equity Capital"]
            reserves = balance_sheet["Reserves"]
            
            if liabilities and equity and reserves and len(liabilities) == len(equity) == len(reserves):
                ratios = []
                for i in range(len(liabilities)):
                    li = safe_float_convert(liabilities[i])
                    eq = safe_float_convert(equity[i])
                    res = safe_float_convert(reserves[i])
                    
                    total_equity = (eq or 0) + (res or 0)
                    if li is not None and total_equity and total_equity > 0:
                        ratios.append(li / total_equity)
                    else:
                        ratios.append(None)
                
                if ratios:
                    current_ratio = ratios[-1] if ratios else None
                    is_low_leverage = all(v and v <= 2.0 for v in ratios if v is not None)
                    
                    return {
                        "dates": dates,
                        "historical_ratios": ratios,
                        "current_ratio": current_ratio,
                        "is_low_leverage": is_low_leverage,
                        "average": sum(v for v in ratios if v is not None) / len([v for v in ratios if v is not None]) if ratios else None
                    }
        
        # For banking entity, we might want to use different metrics like Deposits to Equity or
        # Capital Adequacy Ratio which is more relevant for banks
        if is_banking_entity(ticker) and "Deposits" in balance_sheet and "Equity Capital" in balance_sheet and "Reserves" in balance_sheet:
            deposits = balance_sheet["Deposits"]
            equity = balance_sheet["Equity Capital"]
            reserves = balance_sheet["Reserves"]
            
            if deposits and equity and reserves and len(deposits) == len(equity) == len(reserves):
                ratios = []
                for i in range(len(deposits)):
                    dep = safe_float_convert(deposits[i])
                    eq = safe_float_convert(equity[i])
                    res = safe_float_convert(reserves[i])
                    
                    total_equity = (eq or 0) + (res or 0)
                    if dep is not None and total_equity and total_equity > 0:
                        ratios.append(dep / total_equity)
                    else:
                        ratios.append(None)
                
                if ratios:
                    current_ratio = ratios[-1] if ratios else None
                    # For banks, leverage is naturally higher, so adjusted threshold
                    is_low_leverage = all(v and v <= 10.0 for v in ratios if v is not None)
                    
                    return {
                        "dates": dates,
                        "historical_ratios": ratios,
                        "current_ratio": current_ratio,
                        "is_low_leverage": is_low_leverage,
                        "average": sum(v for v in ratios if v is not None) / len([v for v in ratios if v is not None]) if ratios else None,
                        "source": "deposits_to_equity"
                    }
    
    return {}

def get_free_cash_flow(ticker: str) -> dict:
    """
    Calculate free cash flow (FCF) for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary containing FCF data, including historical values, current value,
        FCF yield, and trend information
    """
    # Try calculating FCF from components: NI + Depreciation - Capex - WC Change
    try:
        # Get required components
        net_income_data = get_net_income(ticker)
        depreciation_data = get_depreciation_amortization(ticker)
        capex_data = get_capital_expenditure(ticker)
        wc_change_data = get_working_capital_change(ticker)
        
        # Check if we have all components for current value
        if (net_income_data and "current_value" in net_income_data and
            depreciation_data and "current_value" in depreciation_data and
            capex_data and "current_value" in capex_data and
            wc_change_data and "current_change" in wc_change_data):
            
            # Calculate FCF
            ni = net_income_data["current_value"]
            dep = depreciation_data["current_value"]
            capex = capex_data["current_value"]
            wc_change = wc_change_data["current_change"]
            
            fcf = ni + dep - capex - wc_change
            
            # Get historical FCF data if possible
            historical_fcf = []
            if (all(k in net_income_data for k in ["dates", "values"]) and
                all(k in depreciation_data for k in ["dates", "values"])):
                
                # Get the minimum length of all component arrays
                min_length = min(
                    len(net_income_data.get("values", [])),
                    len(depreciation_data.get("values", []))
                )
                
                # If we have historical values, calculate historical FCF
                if min_length > 0:
                    for i in range(min_length):
                        hist_ni = net_income_data["values"][i] if i < len(net_income_data["values"]) else None
                        hist_dep = depreciation_data["values"][i] if i < len(depreciation_data["values"]) else None
                        
                        # We may not have historical capex and wc_change values,
                        # so use estimates or current values as approximation
                        if hist_ni is not None and hist_dep is not None:
                            # Use current values as estimates if historical not available
                            estimated_capex = capex
                            estimated_wc_change = wc_change
                            
                            hist_fcf = hist_ni + hist_dep - estimated_capex - estimated_wc_change
                            historical_fcf.append(hist_fcf)
                        else:
                            historical_fcf.append(None)
            
            # Calculate FCF yield if market cap is available
            market_cap = get_market_cap(ticker)
            fcf_yield = None
            if market_cap and market_cap > 0:
                fcf_yield = fcf / market_cap
            
            # Determine FCF trend
            trend = None
            if len(historical_fcf) >= 2 and all(x is not None for x in historical_fcf[-2:]):
                if historical_fcf[-1] > historical_fcf[-2]:
                    trend = "improving"
                elif historical_fcf[-1] < historical_fcf[-2]:
                    trend = "declining"
                else:
                    trend = "stable"
            
            return {
                "historical_fcf": historical_fcf,
                "current_fcf": fcf,
                "fcf_yield": fcf_yield,
                "trend": trend,
                "is_positive": fcf > 0,
                "calculation_method": "component_based"
            }
    except Exception as e:
        print(f"Error calculating FCF from components for {ticker}: {str(e)}")
    
    # If we couldn't calculate from components, try to find FCF directly in cash flows
    try:
        # Get cash flow data
        cash_flows = get_cash_flows(ticker)
        market_cap_value = get_market_cap(ticker)
        
        if cash_flows and "date" in cash_flows:
            dates = cash_flows["date"]
            
            # Look for FCF in direct form
            for field in ["Free Cash Flow", "FCF"]:
                if field in cash_flows:
                    fcf_values = cash_flows[field]
                    
                    if fcf_values and len(fcf_values) > 0:
                        # Convert to float
                        fcf = [safe_float_convert(v) for v in fcf_values]
                        
                        if fcf:
                            current_fcf = fcf[-1] if fcf else None
                            is_positive = any(v and v > 0 for v in fcf if v is not None)
                            
                            # Try to calculate FCF yield if we have market cap
                            fcf_yield = None
                            if market_cap_value and current_fcf and current_fcf > 0:
                                fcf_yield = current_fcf / market_cap_value
                            
                            return {
                                "dates": dates,
                                "historical_fcf": fcf,
                                "current_fcf": current_fcf,
                                "is_positive": is_positive,
                                "fcf_yield": fcf_yield,
                                "average": sum(v for v in fcf if v is not None) / len([v for v in fcf if v is not None]) if fcf else None,
                                "calculation_method": "direct"
                            }
    except Exception as e:
        print(f"Error finding direct FCF for {ticker}: {str(e)}")
        
    # If we still don't have FCF, return None
    return None

def get_cost_of_goods_sold(ticker: str) -> dict:
    """
    Get Cost of Goods Sold (COGS) data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary containing COGS data, including historical values and current value
    """
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data or "profit_loss" not in fundamental_data:
        return {}
    
    profit_loss = fundamental_data["profit_loss"]
    
    # Check various possible field names for COGS
    for field in ["Cost of Goods Sold", "COGS", "Cost of Sales", "Cost of Revenue"]:
        if field in profit_loss:
            cogs_values = profit_loss[field]
            dates = profit_loss.get("date", [])
            
            if cogs_values and len(cogs_values) > 0:
                # Convert to float
                cogs_values = [safe_float_convert(v) for v in cogs_values]
                
                return {
                    "dates": dates,
                    "values": cogs_values,
                    "current_value": cogs_values[0] if cogs_values else None,
                    "is_directly_available": True
                }
    
    # If no direct COGS field, try to calculate from Sales - Gross Profit if available
    if "Sales\xa0+" in profit_loss:
        sales = profit_loss["Sales\xa0+"]
        
        for field in ["Gross Profit"]:
            if field in profit_loss:
                gross_profit = profit_loss[field]
                dates = profit_loss.get("date", [])
                
                if sales and gross_profit and len(sales) == len(gross_profit):
                    # Calculate COGS as Sales - Gross Profit
                    cogs_values = []
                    for s, gp in zip(sales, gross_profit):
                        if s is not None and gp is not None:
                            cogs_values.append(s - gp)
                        else:
                            cogs_values.append(None)
                    
                    return {
                        "dates": dates,
                        "values": cogs_values,
                        "current_value": cogs_values[0] if cogs_values else None,
                        "is_calculated": True,
                        "calculation_method": "sales_minus_gross_profit"
                    }
    
    # If we still don't have COGS, try to estimate from Sales and typical industry margin
    if "Sales\xa0+" in profit_loss:
        sales = profit_loss["Sales\xa0+"]
        dates = profit_loss.get("date", [])
        
        if sales and len(sales) > 0:
            # Assuming a typical gross margin of 40%, COGS would be 60% of sales
            cogs_values = [s * 0.6 if s is not None else None for s in sales]
            
            return {
                "dates": dates,
                "values": cogs_values, 
                "current_value": cogs_values[0] if cogs_values else None,
                "is_estimated": True,
                "estimation_method": "percentage_of_sales"
            }
    
    # Use expenses as a proxy for COGS if nothing else is available
    if "Expenses\xa0+" in profit_loss:
        expenses = profit_loss["Expenses\xa0+"]
        dates = profit_loss.get("date", [])
        
        if expenses and len(expenses) > 0:
            return {
                "dates": dates,
                "values": expenses,
                "current_value": expenses[0] if expenses else None,
                "is_estimated": True,
                "estimation_method": "total_expenses",
                "note": "Using total expenses as a proxy for COGS"
            }
    
    return {}

def get_gross_margin(ticker: str) -> dict:
    """
    Calculate Gross Margin for a ticker from Sales and Cost of Goods Sold.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary containing gross margin data, including historical values and current value
    """
    # Get fundamental data
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data or "profit_loss" not in fundamental_data:
        return {}
    
    profit_loss = fundamental_data["profit_loss"]
    
    # Check if Gross Margin is directly available
    if "Gross Margin %" in profit_loss:
        gross_margin = profit_loss["Gross Margin %"]
        dates = profit_loss.get("date", [])
        
        if gross_margin and len(gross_margin) > 0:
            # Convert percentages to decimals if needed
            gross_margin_values = [safe_float_convert(m) for m in gross_margin]
            gross_margin_values = [m/100 if m and m > 1 else m for m in gross_margin_values]
            
            return {
                "dates": dates,
                "values": gross_margin_values,
                "current_value": gross_margin_values[0] if gross_margin_values else None,
                "is_directly_available": True
            }
    
    # If not directly available, try to calculate from Sales and COGS
    cogs_data = get_cost_of_goods_sold(ticker)
    
    if "Sales\xa0+" in profit_loss and cogs_data and "values" in cogs_data:
        sales = profit_loss["Sales\xa0+"]
        cogs = cogs_data["values"]
        dates = profit_loss.get("date", [])
        
        if sales and cogs and len(sales) == len(cogs):
            # Calculate gross margin: (Sales - COGS) / Sales
            gross_margin_values = []
            for s, c in zip(sales, cogs):
                if s and s != 0 and c is not None:
                    gm = (s - c) / s
                    gross_margin_values.append(gm)
                else:
                    gross_margin_values.append(None)
            
            if gross_margin_values:
                return {
                    "dates": dates,
                    "values": gross_margin_values,
                    "current_value": gross_margin_values[0] if gross_margin_values else None,
                    "is_calculated": True,
                    "calculation_method": "sales_minus_cogs"
                }
    
    # If not available from COGS, try calculating from Sales and Expenses
    if "Sales\xa0+" in profit_loss and "Expenses\xa0+" in profit_loss:
        sales = profit_loss["Sales\xa0+"]
        expenses = profit_loss["Expenses\xa0+"]
        dates = profit_loss.get("date", [])
        
        if sales and expenses and len(sales) == len(expenses):
            # Calculate gross margin: (Sales - Expenses) / Sales
            gross_margin_values = []
            for s, e in zip(sales, expenses):
                if s and s != 0:
                    gm = (1 - e/s)
                    gross_margin_values.append(gm)
                else:
                    gross_margin_values.append(None)
            
            if gross_margin_values:
                return {
                    "dates": dates,
                    "values": gross_margin_values,
                    "current_value": gross_margin_values[0] if gross_margin_values else None,
                    "is_calculated": True,
                    "calculation_method": "sales_minus_expenses"
                }
    
    # If we can't calculate gross margin, try to use operating margin as an approximation
    operating_data = get_operating_margin(ticker)
    if operating_data and "current_operating_margin" in operating_data:
        # Operating margin is usually lower than gross margin
        # Apply a conservative multiplier (typical gross margins are 1.5-2x operating margins)
        operating_margin = operating_data["current_operating_margin"]
        estimated_gross_margin = operating_margin * 1.5
        
        return {
            "dates": operating_data.get("dates", []),
            "values": [m * 1.5 if m is not None else None for m in operating_data.get("values", [])],
            "current_value": estimated_gross_margin,
            "is_estimated": True,
            "estimation_method": "from_operating_margin"
        }
    
    return {}

def get_research_and_development(ticker: str) -> dict:
    """
    Get research and development expenses for a ticker.
    This is a placeholder function for when R&D data is not available.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary containing R&D data with empty values and availability flag set to False
    """
    # Try to find R&D expenses in profit and loss statement
    fundamental_data = fetch_fundamental_data(ticker)
    if fundamental_data and "profit_loss" in fundamental_data:
        profit_loss = fundamental_data["profit_loss"]
        
        # Check various possible R&D field names
        for field in ["Research & Development", "Research and Development", "R&D Expenses", "R&D"]:
            if field in profit_loss:
                rd_values = profit_loss[field]
                dates = profit_loss.get("date", [])
                
                if rd_values and len(rd_values) > 0:
                    # Convert to float
                    rd_values = [safe_float_convert(v) for v in rd_values]
                    
                    # Calculate percentage of revenue if possible
                    percentage_of_revenue = None
                    if "Sales\xa0+" in profit_loss and profit_loss["Sales\xa0+"] and len(profit_loss["Sales\xa0+"]) > 0:
                        sales = profit_loss["Sales\xa0+"][0]
                        if sales and sales != 0 and rd_values[0] is not None:
                            percentage_of_revenue = rd_values[0] / sales
                    
                    return {
                        "historical_values": rd_values,
                        "current_value": rd_values[0] if rd_values else None,
                        "percentage_of_revenue": percentage_of_revenue,
                        "available": True,
                        "dates": dates
                    }
    
    # R&D data is not available in the current fundamental data structure
    # This function is provided as a placeholder to maintain compatibility
    return {
        "historical_values": [],
        "current_value": 0,
        "percentage_of_revenue": 0,
        "available": False,
        "message": "R&D data is not available in the current fundamental data structure"
    }

# Fix recursion issue by defining new functions instead of aliases

def get_net_income(ticker: str) -> Dict[str, Any]:
    """
    Get net income data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with net income data
    """
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data or "profit_loss" not in fundamental_data:
        return {}
    
    profit_loss = fundamental_data["profit_loss"]
    
    # Look for net income in different key formats
    for income_key in ["Net Profit\xa0+", "Net Income", "Net Profit", "Profit After Tax"]:
        if income_key in profit_loss:
            income_values = profit_loss[income_key]
            income_values = [safe_float_convert(v) for v in income_values]
            valid_values = [v for v in income_values if v is not None]
            
            if valid_values:
                current_value = valid_values[0]
                return {
                    "dates": profit_loss.get("date", []),
                    "values": income_values,
                    "current_value": current_value,
                    "average": sum(valid_values) / len(valid_values) if valid_values else None,
                    "trend": "increasing" if len(valid_values) > 1 and valid_values[0] > valid_values[-1] else "decreasing"
                }
    
    return {}

def get_depreciation_amortization(ticker: str) -> Dict[str, Any]:
    """
    Get depreciation and amortization data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with depreciation/amortization data
    """
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {}
    
    # Try to get from profit_loss
    if "profit_loss" in fundamental_data:
        profit_loss = fundamental_data["profit_loss"]
        for dep_key in ["Depreciation", "Depreciation & Amortization", "Depreciation and Amortization"]:
            if dep_key in profit_loss:
                dep_values = profit_loss[dep_key]
                dep_values = [safe_float_convert(v) for v in dep_values]
                valid_values = [v for v in dep_values if v is not None]
                
                if valid_values:
                    current_value = valid_values[0]
                    return {
                        "dates": profit_loss.get("date", []),
                        "values": dep_values,
                        "current_value": current_value,
                        "average": sum(valid_values) / len(valid_values) if valid_values else None
                    }
    
    # Look for depreciation in the profit/loss statement
    if "profit_loss" in fundamental_data and "Depreciation" in fundamental_data["profit_loss"]:
        depreciation = fundamental_data["profit_loss"]["Depreciation"]
        if depreciation and len(depreciation) > 0:
            dep_value = safe_float_convert(depreciation[0])
            if dep_value is not None:
                return {
                    "current_value": dep_value,
                    "values": [safe_float_convert(v) for v in depreciation],
                    "dates": fundamental_data["profit_loss"].get("date", [])
                }
    
    # If no direct data, estimate based on typical depreciation rates
    if "balance_sheet" in fundamental_data and "Fixed Assets\xa0+" in fundamental_data["balance_sheet"]:
        fixed_assets = fundamental_data["balance_sheet"]["Fixed Assets\xa0+"]
        if fixed_assets and len(fixed_assets) > 0:
            fa_value = safe_float_convert(fixed_assets[0])
            if fa_value is not None:
                # Assume depreciation is ~10% of fixed assets
                return {
                    "current_value": fa_value * 0.1,
                    "is_estimated": True,
                    "estimation_method": "percentage_of_fixed_assets"
                }
    
    return {}

def get_capital_expenditure(ticker: str) -> Dict[str, Any]:
    """
    Get capital expenditure data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with capital expenditure data
    """
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {}
    
    # First try to find CapEx in cash flows if available
    if "cash_flows" in fundamental_data:
        cash_flows = fundamental_data["cash_flows"]
        for capex_key in ["Capital Expenditure", "CAPEX", "Purchase of Fixed Assets"]:
            if capex_key in cash_flows:
                capex_values = cash_flows[capex_key]
                capex_values = [safe_float_convert(v) for v in capex_values]
                valid_values = [v for v in capex_values if v is not None]
                
                if valid_values:
                    current_value = valid_values[0]
                    # Ensure capex is expressed as a positive value for consistency
                    current_value = abs(current_value)
                    return {
                        "dates": cash_flows.get("date", []),
                        "values": [abs(v) if v is not None else None for v in capex_values],
                        "current_value": current_value,
                        "average": sum(abs(v) for v in valid_values) / len(valid_values) if valid_values else None
                    }
    
    # If no direct capex found, try to estimate from fixed assets change
    if "balance_sheet" in fundamental_data:
        balance_sheet = fundamental_data["balance_sheet"]
        fixed_asset_keys = ["Fixed Assets\xa0+", "Fixed Assets", "Property, Plant & Equipment"]
        
        for fa_key in fixed_asset_keys:
            if fa_key in balance_sheet:
                fa_values = balance_sheet[fa_key]
                fa_values = [safe_float_convert(v) for v in fa_values]
                valid_values = [v for v in fa_values if v is not None]
                
                if len(valid_values) >= 2:
                    # Estimate CapEx as year-over-year change in fixed assets
                    # Plus estimated depreciation (if available)
                    current_fa = valid_values[0]
                    previous_fa = valid_values[1]
                    fa_change = current_fa - previous_fa
                    
                    # Get depreciation if available
                    depreciation = 0
                    dep_data = get_depreciation_amortization(ticker)
                    if dep_data and "current_value" in dep_data:
                        depreciation = dep_data["current_value"]
                    
                    # CapEx = Change in Fixed Assets + Depreciation
                    estimated_capex = max(0, fa_change + depreciation)
                    
                    return {
                        "dates": balance_sheet.get("date", []),
                        "current_value": estimated_capex,
                        "is_estimated": True,
                        "estimation_method": "fixed_assets_change_plus_depreciation"
                    }
    
    # If all else fails, estimate as a percentage of net income or revenue
    # This is a very rough estimate
    ni_data = get_net_income(ticker)
    if ni_data and "current_value" in ni_data:
        current_ni = ni_data["current_value"]
        # Assume CapEx is roughly 20-30% of net income for mature companies
        estimated_capex = current_ni * 0.25
        
        return {
            "current_value": estimated_capex,
            "is_estimated": True,
            "estimation_method": "percentage_of_net_income"
        }
    
    # Try revenue-based estimation as last resort
    if "profit_loss" in fundamental_data:
        profit_loss = fundamental_data["profit_loss"]
        revenue_key = None
        for key in ["Sales\xa0+", "Revenue", "Total Revenue"]:
            if key in profit_loss:
                revenue_key = key
                break
        
        if revenue_key and profit_loss[revenue_key] and len(profit_loss[revenue_key]) > 0:
            revenue = safe_float_convert(profit_loss[revenue_key][0])
            if revenue is not None:
                # CapEx is typically 5-10% of revenue
                return {
                    "current_value": revenue * 0.07,
                    "is_estimated": True,
                    "estimation_method": "percentage_of_revenue"
                }
    
    return {}

def get_working_capital_change(ticker: str) -> Dict[str, Any]:
    """
    Get working capital change data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with working capital change data
    """
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {}
    
    # Try to calculate from balance sheet
    if "balance_sheet" in fundamental_data:
        balance_sheet = fundamental_data["balance_sheet"]
        
        # Check for current assets and current liabilities
        current_assets_key = None
        for key in ["Current Assets", "Other Assets\xa0+"]:
            if key in balance_sheet:
                current_assets_key = key
                break
        
        current_liabilities_key = None
        for key in ["Current Liabilities", "Other Liabilities\xa0+"]:
            if key in balance_sheet:
                current_liabilities_key = key
                break
        
        if current_assets_key and current_liabilities_key:
            current_assets = balance_sheet[current_assets_key]
            current_liabilities = balance_sheet[current_liabilities_key]
            
            if len(current_assets) >= 2 and len(current_liabilities) >= 2:
                # Current working capital
                current_ca = safe_float_convert(current_assets[0])
                current_cl = safe_float_convert(current_liabilities[0])
                
                # Previous working capital
                previous_ca = safe_float_convert(current_assets[1])
                previous_cl = safe_float_convert(current_liabilities[1])
                
                if all(v is not None for v in [current_ca, current_cl, previous_ca, previous_cl]):
                    current_wc = current_ca - current_cl
                    previous_wc = previous_ca - previous_cl
                    wc_change = current_wc - previous_wc
                    
                    return {
                        "current_change": wc_change,
                        "current_wc": current_wc,
                        "previous_wc": previous_wc
                    }
    
    # If we have working capital days, try to use that
    if "cash_flows" in fundamental_data and "Working Capital Days" in fundamental_data["cash_flows"]:
        wc_days = fundamental_data["cash_flows"]["Working Capital Days"]
        if len(wc_days) >= 2:
            current_days = safe_float_convert(wc_days[0])
            previous_days = safe_float_convert(wc_days[1])
            
            # Need revenue to estimate working capital in currency terms
            if "profit_loss" in fundamental_data:
                profit_loss = fundamental_data["profit_loss"]
                revenue_key = None
                for key in ["Sales\xa0+", "Revenue", "Total Revenue"]:
                    if key in profit_loss:
                        revenue_key = key
                        break
                
                if revenue_key and profit_loss[revenue_key] and len(profit_loss[revenue_key]) > 0:
                    revenue = safe_float_convert(profit_loss[revenue_key][0])
                    if revenue is not None and current_days is not None and previous_days is not None:
                        # WC = WC Days * (Revenue/365)
                        daily_revenue = revenue / 365
                        current_wc = current_days * daily_revenue
                        previous_wc = previous_days * daily_revenue
                        wc_change = current_wc - previous_wc
                        
                        return {
                            "current_change": wc_change,
                            "current_wc_days": current_days,
                            "previous_wc_days": previous_days,
                            "daily_revenue": daily_revenue
                        }
    
    # Default fallback - estimate as small percentage of net income
    ni_data = get_net_income(ticker)
    if ni_data and "current_value" in ni_data:
        ni = ni_data["current_value"]
        return {
            "current_change": ni * 0.05,  # Assume 5% of net income
            "is_estimated": True,
            "estimation_method": "percentage_of_net_income"
        }
    
    return {}

def get_earnings_growth(ticker: str) -> Dict[str, Any]:
    """
    Get earnings growth data for a ticker.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with earnings growth data
    """
    fundamental_data = fetch_fundamental_data(ticker)
    if not fundamental_data:
        return {"growth_rate": 0.05}  # Default conservative growth rate
    
    # Try to get from compounded profit growth
    if "compounded_profit_growth" in fundamental_data:
        profit_growth = fundamental_data["compounded_profit_growth"]
        if "Compounded Profit Growth" in profit_growth:
            growth_values = profit_growth["Compounded Profit Growth"]
            if growth_values and len(growth_values) > 0:
                growth_rate = safe_float_convert(growth_values[0])
                if growth_rate is not None:
                    # Convert percentage to decimal if needed
                    if growth_rate > 1:
                        growth_rate = growth_rate / 100
                    
                    return {
                        "growth_rate": growth_rate,
                        "source": "compounded_profit_growth"
                    }
    
    # Try to calculate from profit history
    ni_data = get_net_income(ticker)
    if ni_data and "values" in ni_data:
        values = ni_data["values"]
        valid_values = [v for v in values if v is not None]
        
        if len(valid_values) >= 2:
            start_value = valid_values[-1]  # Oldest value
            end_value = valid_values[0]    # Newest value
            years = len(valid_values) - 1
            
            if start_value > 0:
                # Calculate CAGR: (End/Start)^(1/Years) - 1
                cagr = (end_value / start_value) ** (1 / years) - 1
                return {
                    "growth_rate": cagr,
                    "source": "calculated_from_net_income",
                    "start_value": start_value,
                    "end_value": end_value,
                    "years": years
                }
    
    # Try revenue growth as fallback
    if "compounded_sales_growth" in fundamental_data:
        sales_growth = fundamental_data["compounded_sales_growth"]
        if "Compounded Sales Growth" in sales_growth:
            growth_values = sales_growth["Compounded Sales Growth"]
            if growth_values and len(growth_values) > 0:
                growth_rate = safe_float_convert(growth_values[0])
                if growth_rate is not None:
                    # Convert percentage to decimal if needed
                    if growth_rate > 1:
                        growth_rate = growth_rate / 100
                    
                    # Revenue growth is typically less than profit growth
                    # so we'll add a small premium
                    return {
                        "growth_rate": growth_rate * 1.1,
                        "source": "sales_growth_with_premium"
                    }
    
    # Industry average as last resort
    return {
        "growth_rate": 0.05,  # 5% as conservative default
        "source": "industry_average_fallback"
    }

def get_research_and_development(ticker: str) -> Dict[str, Any]:
    """
    Get research and development data for a ticker.
    This is a placeholder function that returns empty data since R&D information
    is not available in our current fundamental data structure.
    
    Args:
        ticker: The ticker symbol
        
    Returns:
        Dictionary with empty R&D data
    """
    # Since we don't have R&D data in our fundamental data structure,
    # this function returns an empty result that won't break code that calls it
    return {
        "historical_values": [],
        "current_value": 0,
        "pct_of_revenue": 0,
        "is_available": False
    }

