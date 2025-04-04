import asyncio
import csv
import json
import os
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

from bs4 import BeautifulSoup, Tag
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode


async def scrape_stock_data(crawler: AsyncWebCrawler, symbol: str) -> Dict[str, Any]:
    """
    Scrape financial data for a single stock from screener.in
    
    Args:
        crawler: AsyncWebCrawler instance
        symbol: Stock symbol to scrape data for
        
    Returns:
        Dictionary containing all the scraped financial data
    """
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    print(f"Scraping data for {symbol} from {url}")
    
    try:
        # Configure crawler with appropriate content selection and filtering
        config = CrawlerRunConfig(
            # Target specific content sections on screener.in
            target_elements=[
                "#quarterly-results",  # Quarterly Results section
                "#profit-loss",       # Profit & Loss section
                "#balance-sheet",     # Balance Sheet section
                "#cash-flow",         # Cash Flow section
                "#ratios",            # Ratios section
                "#shareholding",      # Shareholding Pattern section
                ".company-ratios",    # Company ratios with growth metrics
                ".company-info",      # Company information
                "#peer-comparison"    # Peer comparison section (may contain growth metrics)
            ],
            # Filter out unnecessary elements
            excluded_tags=["nav", "footer", "header", "form", "script", "style"],
            # Set minimum word count to filter out trivial text blocks
            word_count_threshold=2,  # Lowered to catch more data
            # Exclude external links and images to focus on the data
            exclude_external_links=True,
            exclude_external_images=True,
            # Use fresh data
            cache_mode=CacheMode.BYPASS
        )
        
        # Run the crawler with the configured settings
        result = await crawler.arun(
            url=url,
            config=config,
            save_html=True,     # Save the HTML content
            save_markdown=True, # Save the markdown content
            save_text=True      # Save the text content
        )
        
        # Create a structured data dictionary
        stock_data = {
            "symbol": symbol,
            "url": url,
            "timestamp": time.time(),
            "quarterly_results": {},
            "profit_loss": {},
            "compounded_sales_growth": {},
            "compounded_profit_growth": {},
            "stock_price_cagr": {},
            "return_on_equity": {},
            "balance_sheet": {},
            "cash_flows": {},
            "ratios": {},
            "shareholding_pattern": {},
            "market_data": {},
            "remarks": {
                "pros": [],
                "cons": []
            }
        }
        
        # Process the data
        if hasattr(result, 'html') and result.html:
            # Extract data from HTML
            soup = BeautifulSoup(result.html, 'html.parser')
            # Extract tables from the HTML
            tables = extract_tables_from_html(soup)
            
            # Process each table based on its section ID
            for section_id, table_data in tables.items():
                if any(term in section_id.lower() for term in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]):
                    stock_data["quarterly_results"] = table_data
                elif any(term in section_id.lower() for term in ["profit", "loss", "p&l", "p and l"]):
                    stock_data["profit_loss"] = table_data
                elif any(term in section_id.lower() for term in ["balance", "sheet", "bs", "assets", "liabilities"]):
                    stock_data["balance_sheet"] = table_data
                elif any(term in section_id.lower() for term in ["cash", "flow", "cf", "cash flows"]):
                    stock_data["cash_flows"] = table_data
                elif any(term in section_id.lower() for term in ["ratio", "ratios", "financial ratios"]):
                    stock_data["ratios"] = table_data
                elif any(term in section_id.lower() for term in ["shareholding", "holding", "share holding", "ownership"]):
                    stock_data["shareholding_pattern"] = table_data
            
            # Extract growth metrics and other key indicators
            metrics = extract_metrics_from_html(soup)
            for metric_name, metric_data in metrics.items():
                if metric_name in stock_data:
                    stock_data[metric_name] = metric_data
                    
            # Make sure market data is included
            if "market_data" in metrics and metrics["market_data"]:
                stock_data["market_data"] = metrics["market_data"]
                    
            # Extract pros and cons
            remarks = extract_remarks_from_html(soup)
            if remarks:
                stock_data["remarks"] = remarks
        
        # If we have markdown content, use it as a fallback
        if hasattr(result, 'markdown'):
            # Extract data from markdown as a fallback for any missing sections
            sections = extract_sections_from_markdown(result.markdown)
            
            # Populate the stock_data dictionary with extracted information
            for section_name, section_data in sections.items():
                if section_name in stock_data and (not stock_data[section_name] or len(stock_data[section_name]) == 0):
                    stock_data[section_name] = section_data
        
        # Save raw data for debugging (only in test mode)
        if hasattr(result, 'markdown'):
            stock_data["raw_markdown"] = result.markdown
        if hasattr(result, 'html'):
            stock_data["raw_html"] = result.html[:10000]  # Limit size to avoid huge files
        if hasattr(result, 'text'):
            stock_data["raw_text"] = result.text
        
        # Post-process the data to ensure all tables have consistent structure
        stock_data = post_process_data(stock_data)
        
        return stock_data
        
    except Exception as e:
        print(f"Error scraping data for {symbol}: {str(e)}")
        return {
            "symbol": symbol,
            "url": url,
            "error": str(e),
            "timestamp": time.time()
        }


def extract_tables_from_html(soup: BeautifulSoup) -> Dict[str, Dict[str, Any]]:
    """
    Extract tables from HTML content using BeautifulSoup
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Dictionary mapping section IDs to table data
    """
    tables_data = {}
    
    # First, try to find specific sections by their IDs or headings
    section_identifiers = {
        "quarterly_results": ["quarterly", "quarterly results", "quarterly financials"],
        "profit_loss": ["profit", "loss", "profit & loss", "profit and loss", "p&l"],
        "balance_sheet": ["balance", "balance sheet", "assets", "liabilities"],
        "cash_flows": ["cash", "cash flow", "cash flows"],
        "ratios": ["ratio", "ratios", "financial ratios"],
        "shareholding_pattern": ["shareholding", "holding", "shareholding pattern"]
    }
    
    # Find all sections, divs, or elements that might contain tables
    potential_sections = []
    
    # Look for section elements
    potential_sections.extend(soup.find_all('section'))
    
    # Look for divs with specific classes or IDs
    potential_sections.extend(soup.find_all('div', class_=lambda c: c and any(x in (c.lower() if c else '') for x in 
                                                                           ['section', 'card', 'table-responsive', 'quarterly', 'profit', 'balance', 'cash', 'ratio', 'shareholding'])))
    
    # Look for divs with headings that match our sections
    for div in soup.find_all('div'):
        heading = div.find(['h1', 'h2', 'h3', 'h4'])
        if heading and any(term in heading.text.lower() for terms in section_identifiers.values() for term in terms):
            potential_sections.append(div)
    
    # Process each potential section
    for section in potential_sections:
        # Try to identify the section type
        section_type = None
        section_text = section.text.lower()
        
        # Check section ID
        if section.get('id'):
            section_id = section['id'].lower()
            for s_type, terms in section_identifiers.items():
                if any(term in section_id for term in terms):
                    section_type = s_type
                    break
        
        # Check section heading if type not identified yet
        if not section_type:
            heading = section.find(['h1', 'h2', 'h3', 'h4'])
            if heading:
                heading_text = heading.text.lower()
                for s_type, terms in section_identifiers.items():
                    if any(term in heading_text for term in terms):
                        section_type = s_type
                        break
        
        # Check section text content if type not identified yet
        if not section_type:
            for s_type, terms in section_identifiers.items():
                if any(term in section_text for term in terms):
                    section_type = s_type
                    break
        
        # If we still couldn't identify the section type, generate a generic ID
        if not section_type:
            # Try to use any heading text as the section ID
            heading = section.find(['h1', 'h2', 'h3', 'h4'])
            if heading:
                section_type = heading.text.strip().lower().replace(' ', '_')
            else:
                # Generate a hash-based ID as a last resort
                section_type = f"section_{hash(section_text[:50])}"
        
        # Find table in this section
        table = section.find('table')
        if not table:
            continue
        
        # Extract table headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all('th')]
        else:
            # Try to get headers from the first row if no thead
            first_row = table.find('tr')
            if first_row:
                headers = [th.text.strip() for th in first_row.find_all(['th', 'td'])]
        
        # Extract table rows
        rows = []
        body_rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:] if headers else table.find_all('tr')
        
        for row in body_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == 0:
                continue
            
            # If headers are missing or don't match cell count, generate appropriate headers
            if not headers or len(headers) != len(cells):
                # For financial tables, first column is often the metric name
                if len(cells) > 1:
                    # First column is likely the metric name, rest are time periods
                    headers = ['Metric'] + [f"Period {i}" for i in range(1, len(cells))]
                else:
                    headers = [f"Column {i+1}" for i in range(len(cells))]
            
            # Create a dictionary for this row
            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    header = headers[i]
                    # Clean the cell text
                    cell_text = cell.text.strip()
                    row_data[header] = cell_text
            
            if row_data:
                rows.append(row_data)
        
        # Store the table data if we found any rows
        if rows:
            tables_data[section_type] = {
                "headers": headers,
                "rows": rows
            }
    
    # Special handling for growth metrics which might be in separate elements
    growth_metrics = extract_growth_metrics(soup)
    if growth_metrics:
        tables_data.update(growth_metrics)
    
    return tables_data


def extract_remarks_from_html(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    Extract pros and cons from HTML content
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Dictionary with pros and cons lists
    """
    remarks = {
        "pros": [],
        "cons": []
    }
    
    # First approach: Look for specific sections with PROS and CONS headings
    pros_heading = soup.find(string=lambda s: s and 'PROS' in s)
    cons_heading = soup.find(string=lambda s: s and 'CONS' in s)
    
    # If we found the headings, look for the list items
    if pros_heading:
        # Navigate up to find a container
        pros_container = pros_heading
        for _ in range(3):  # Try a few levels up
            if pros_container.parent:
                pros_container = pros_container.parent
            else:
                break
        
        # Look for list items
        pros_items = pros_container.find_all('li')
        if pros_items:
            for item in pros_items:
                text = item.text.strip()
                if text and not text.lower().startswith('cons'):
                    remarks["pros"].append(text)
    
    if cons_heading:
        # Navigate up to find a container
        cons_container = cons_heading
        for _ in range(3):  # Try a few levels up
            if cons_container.parent:
                cons_container = cons_container.parent
            else:
                break
        
        # Look for list items
        cons_items = cons_container.find_all('li')
        if cons_items:
            for item in cons_items:
                text = item.text.strip()
                if text and not text.lower().startswith('pros'):
                    remarks["cons"].append(text)
    
    # Second approach: Look for sections with class names related to pros/cons
    if not remarks["pros"] and not remarks["cons"]:
        # Try to find sections by class or id
        pros_section = soup.find(class_=lambda c: c and any(term in c.lower() for term in ['pros', 'positive', 'advantage']))
        cons_section = soup.find(class_=lambda c: c and any(term in c.lower() for term in ['cons', 'negative', 'disadvantage']))
        
        if pros_section:
            pros_items = pros_section.find_all('li')
            if pros_items:
                for item in pros_items:
                    remarks["pros"].append(item.text.strip())
        
        if cons_section:
            cons_items = cons_section.find_all('li')
            if cons_items:
                for item in cons_items:
                    remarks["cons"].append(item.text.strip())
    
    # Third approach: Look for the specific structure shown in the screenshot
    if not remarks["pros"] and not remarks["cons"]:
        # Find all list items with bullet points
        all_list_items = soup.find_all('li')
        
        # Group them by their parent to identify pros and cons sections
        parent_to_items = {}
        for item in all_list_items:
            parent = item.parent
            if parent not in parent_to_items:
                parent_to_items[parent] = []
            parent_to_items[parent].append(item)
        
        # Check each group of items
        for parent, items in parent_to_items.items():
            # Check if this is a pros section
            parent_text = parent.get_text().lower()
            if 'pros' in parent_text and not remarks["pros"]:
                for item in items:
                    if 'company has' in item.text.lower() or 'company is' in item.text.lower():
                        remarks["pros"].append(item.text.strip())
            # Check if this is a cons section
            elif 'cons' in parent_text and not remarks["cons"]:
                for item in items:
                    if 'stock is' in item.text.lower():
                        remarks["cons"].append(item.text.strip())
    
    # Fourth approach: Parse the raw HTML to find the pros and cons
    if not remarks["pros"] and not remarks["cons"] and hasattr(soup, 'prettify'):
        html_text = soup.prettify()
        
        # Look for patterns in the HTML that might indicate pros and cons
        pros_pattern = r'PROS[\s\S]*?<li>(.*?)</li>'
        cons_pattern = r'CONS[\s\S]*?<li>(.*?)</li>'
        
        pros_matches = re.findall(pros_pattern, html_text)
        cons_matches = re.findall(cons_pattern, html_text)
        
        if pros_matches:
            for match in pros_matches:
                clean_text = re.sub(r'<.*?>', '', match).strip()
                if clean_text:
                    remarks["pros"].append(clean_text)
        
        if cons_matches:
            for match in cons_matches:
                clean_text = re.sub(r'<.*?>', '', match).strip()
                if clean_text:
                    remarks["cons"].append(clean_text)
    
    # Fifth approach: Extract from the raw markdown if available
    if (not remarks["pros"] or not remarks["cons"]) and hasattr(soup, '_source') and soup._source:
        markdown_text = soup._source
        
        # Look for pros and cons sections in markdown
        pros_section_match = re.search(r'PROS\s*(.*?)\s*CONS', markdown_text, re.DOTALL)
        cons_section_match = re.search(r'CONS\s*(.*?)\s*(?:\n\n|$)', markdown_text, re.DOTALL)
        
        if pros_section_match and not remarks["pros"]:
            pros_text = pros_section_match.group(1)
            pros_items = re.findall(r'[•\*-]\s*(.*?)\s*(?:[•\*-]|$)', pros_text)
            if pros_items:
                remarks["pros"] = [item.strip() for item in pros_items if item.strip()]
        
        if cons_section_match and not remarks["cons"]:
            cons_text = cons_section_match.group(1)
            cons_items = re.findall(r'[•\*-]\s*(.*?)\s*(?:[•\*-]|$)', cons_text)
            if cons_items:
                remarks["cons"] = [item.strip() for item in cons_items if item.strip()]
    
    return remarks


def extract_growth_metrics(soup: BeautifulSoup) -> Dict[str, Dict[str, Any]]:
    """
    Extract growth metrics that might be in separate elements rather than tables
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Dictionary with growth metrics data
    """
    metrics_data = {}
    
    # Look for growth metrics sections
    growth_sections = soup.find_all(['div', 'section'], class_=lambda c: c and any(x in (c.lower() if c else '') for x in 
                                                                                 ['growth', 'metrics', 'cagr', 'compounded', 'return']))
    
    # Also look for specific metric labels
    metric_labels = {
        "compounded_sales_growth": ["compounded sales growth", "sales growth", "revenue growth"],
        "compounded_profit_growth": ["compounded profit growth", "profit growth", "net profit growth"],
        "stock_price_cagr": ["stock price cagr", "price cagr", "share price cagr"],
        "return_on_equity": ["return on equity", "roe", "return on shareholders' equity"]
    }
    
    # Process each growth section
    for section in growth_sections:
        section_text = section.text.lower()
        
        # Check for each metric
        for metric_key, labels in metric_labels.items():
            if any(label in section_text for label in labels):
                # Try to extract the metric value
                for label in labels:
                    if label in section_text:
                        # Find the value after the label
                        parts = section_text.split(label)
                        if len(parts) > 1:
                            value_part = parts[1].strip()
                            match = re.search(r'[-+]?\d+\.?\d*\s*%?', value_part)
                            if match:
                                value = match.group(0).strip()
                                # Store as a structured table-like format for consistency
                                metrics_data[metric_key] = {
                                    "headers": ["Metric", "Value"],
                                    "rows": [{"Metric": label.title(), "Value": value}]
                                }
                                break
    
    return metrics_data


def post_process_data(stock_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the extracted data to ensure consistent structure with appropriate data types
    and convert to a more concise format with dates and values as lists
    
    Args:
        stock_data: The extracted stock data
        
    Returns:
        Processed stock data with consistent structure and proper data types
    """
    # Process each table section
    table_sections = [
        "quarterly_results", "profit_loss", "balance_sheet", 
        "cash_flows", "ratios", "shareholding_pattern"
    ]
    
    for section in table_sections:
        if section in stock_data and isinstance(stock_data[section], dict):
            # Ensure headers and rows exist
            if "headers" not in stock_data[section]:
                stock_data[section]["headers"] = []
            if "rows" not in stock_data[section]:
                stock_data[section]["rows"] = []
            
            # If we have rows but no headers, try to generate headers from the first row
            if not stock_data[section]["headers"] and stock_data[section]["rows"]:
                if len(stock_data[section]["rows"]) > 0:
                    first_row = stock_data[section]["rows"][0]
                    stock_data[section]["headers"] = list(first_row.keys())
            
            # Convert to new format with a shared date array for all metrics
            if "headers" in stock_data[section] and "rows" in stock_data[section]:
                headers = stock_data[section]["headers"]
                rows = stock_data[section]["rows"]
                
                # Create a new structure
                new_structure = {}
                
                # Extract dates from headers (skip the first one as it's usually empty or a label)
                dates = headers[1:] if len(headers) > 1 else []
                
                # Add the shared date array
                new_structure["date"] = dates
                
                # Process each row
                for row in rows:
                    # Get the metric name (usually in the first column)
                    metric_name = row.get("", "")
                    if isinstance(metric_name, str):
                        metric_name = metric_name.strip()
                    if not metric_name:
                        continue
                    
                    # Create list for values
                    value_list = []
                    
                    # Add values for each date
                    for date in dates:
                        if date in row:
                            # Convert value to appropriate type (int or float)
                            value = row[date]
                            if isinstance(value, str):
                                # Remove any non-numeric characters except for decimal point and negative sign
                                clean_value = value.replace(",", "").replace("%", "").strip()
                                try:
                                    # Try to convert to float first
                                    if "." in clean_value:
                                        value = float(clean_value)
                                    else:
                                        value = int(clean_value) if clean_value else 0
                                except ValueError:
                                    # If conversion fails, keep as string
                                    pass
                            
                            # Add to values list
                            value_list.append(value)
                        else:
                            # If date is missing, add null value to maintain alignment
                            value_list.append(None)
                    
                    # Add this metric to the new structure
                    new_structure[metric_name] = value_list
                
                # Replace the old structure with the new one
                stock_data[section] = new_structure
    
    # Process growth metrics
    metric_sections = [
        "compounded_sales_growth", "compounded_profit_growth", 
        "stock_price_cagr", "return_on_equity"
    ]
    
    for section in metric_sections:
        if section in stock_data:
            # If it's a dictionary with a value key, convert to structured format
            if isinstance(stock_data[section], dict):
                # If it has the old format with headers and rows
                if "headers" in stock_data[section] and "rows" in stock_data[section]:
                    rows = stock_data[section]["rows"]
                    periods = []
                    values = []
                    
                    for row in rows:
                        period = row.get("Period", "N/A")
                        value = row.get("Value", 0)
                        
                        # Convert value to appropriate type
                        if isinstance(value, str):
                            clean_value = value.replace(",", "").replace("%", "").strip()
                            try:
                                if "." in clean_value:
                                    value = float(clean_value)
                                else:
                                    value = int(clean_value) if clean_value else 0
                            except ValueError:
                                pass
                        
                        periods.append(period)
                        values.append(value)
                    
                    # Use a shared date array for all metrics
                    stock_data[section] = {
                        "date": periods
                    }
                    stock_data[section][section.replace("_", " ").title()] = values
                # If it has a simple value key
                elif "value" in stock_data[section]:
                    value = stock_data[section]["value"]
                    period = stock_data[section].get("period", "N/A")
                    
                    # Convert value to appropriate type
                    if isinstance(value, str):
                        clean_value = value.replace(",", "").replace("%", "").strip()
                        try:
                            if "." in clean_value:
                                value = float(clean_value)
                            else:
                                value = int(clean_value) if clean_value else 0
                        except ValueError:
                            pass
                    
                    # Use a shared date array for all metrics
                    stock_data[section] = {
                        "date": [period]
                    }
                    stock_data[section][section.replace("_", " ").title()] = [value]
                # If it's already in a dictionary format but not list-based
                elif not ("date" in stock_data[section] and all(k != "date" and isinstance(v, list) for k, v in stock_data[section].items())):
                    periods = []
                    values = []
                    
                    for period, value in stock_data[section].items():
                        if isinstance(value, str):
                            clean_value = value.replace(",", "").replace("%", "").strip()
                            try:
                                if "." in clean_value:
                                    value = float(clean_value)
                                else:
                                    value = int(clean_value) if clean_value else 0
                            except ValueError:
                                pass
                        
                        periods.append(period)
                        values.append(value)
                    
                    # Use a shared date array for all metrics
                    stock_data[section] = {
                        "date": periods
                    }
                    stock_data[section][section.replace("_", " ").title()] = values
    
    return stock_data


def extract_market_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract market data like Market Cap, Current Price, etc.
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Dictionary with market data
    """
    market_data = {}
    
    # Look for market data in specific sections
    market_sections = soup.find_all(['div', 'section'], class_=lambda c: c and any(x in (c.lower() if c else '') for x in 
                                                                           ['company-info', 'market-data', 'stock-info', 'about']))
    
    # Define patterns to look for
    patterns = {
        'Market Cap': [r'market\s+cap\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*cr', r'market\s+cap\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
        'Current Price': [r'current\s+price\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
        'High / Low': [r'high\s*/\s*low\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*/\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
        'Stock P/E': [r'stock\s+p/?e\s*[:\s]\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
        'Book Value': [r'book\s+value\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
        'Dividend Yield': [r'dividend\s+yield\s*[:\s]\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*%'],
        'ROCE': [r'roce\s*[:\s]\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*%'],
        'ROE': [r'roe\s*[:\s]\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*%'],
        'Face Value': [r'face\s+value\s*[:\s]\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'],
    }
    
    # First try to find market data in dedicated sections
    for section in market_sections:
        section_text = section.text.lower()
        
        # Look for each market data item
        for key, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                matches = re.findall(pattern, section_text)
                if matches:
                    # Clean and convert the value
                    if isinstance(matches[0], tuple):  # For high/low pattern
                        high, low = matches[0]
                        high_val = high.replace(',', '')
                        low_val = low.replace(',', '')
                        try:
                            if '.' in high_val:
                                high_val = float(high_val)
                            else:
                                high_val = int(high_val)
                            
                            if '.' in low_val:
                                low_val = float(low_val)
                            else:
                                low_val = int(low_val)
                        except ValueError:
                            pass
                        
                        market_data[key] = f"{high_val} / {low_val}"
                    else:  # For other patterns
                        value = matches[0].replace(',', '')
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass
                        
                        market_data[key] = value
                    break
    
    # If we didn't find all market data in dedicated sections, look for them anywhere
    if len(market_data) < len(patterns):
        all_text = soup.text.lower()
        
        for key, regex_patterns in patterns.items():
            if key not in market_data:
                for pattern in regex_patterns:
                    matches = re.findall(pattern, all_text)
                    if matches:
                        # Clean and convert the value
                        if isinstance(matches[0], tuple):  # For high/low pattern
                            high, low = matches[0]
                            high_val = high.replace(',', '')
                            low_val = low.replace(',', '')
                            try:
                                if '.' in high_val:
                                    high_val = float(high_val)
                                else:
                                    high_val = int(high_val)
                                
                                if '.' in low_val:
                                    low_val = float(low_val)
                                else:
                                    low_val = int(low_val)
                            except ValueError:
                                pass
                            
                            market_data[key] = f"{high_val} / {low_val}"
                        else:  # For other patterns
                            value = matches[0].replace(',', '')
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass
                            
                            market_data[key] = value
                        break
    
    # Look for key-value pairs in table-like structures
    if len(market_data) < len(patterns):
        # Find all elements that might contain key-value pairs
        key_value_elements = soup.find_all(['tr', 'div'], class_=lambda c: c and any(x in (c.lower() if c else '') for x in 
                                                                                 ['row', 'item', 'data', 'stat']))
        
        for element in key_value_elements:
            # Check if this element has two child elements (key and value)
            children = element.find_all(['td', 'div', 'span'])
            if len(children) >= 2:
                key_text = children[0].text.strip().lower()
                value_text = children[1].text.strip().lower()
                
                # Check for each market data key
                for key in patterns.keys():
                    if key.lower() in key_text:
                        # Extract numeric value
                        if key == 'High / Low':
                            value_match = re.search(r'₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*/\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)', value_text)
                            if value_match:
                                high = value_match.group(1).replace(',', '')
                                low = value_match.group(2).replace(',', '')
                                try:
                                    if '.' in high:
                                        high = float(high)
                                    else:
                                        high = int(high)
                                    
                                    if '.' in low:
                                        low = float(low)
                                    else:
                                        low = int(low)
                                except ValueError:
                                    pass
                                
                                market_data[key] = f"{high} / {low}"
                        else:
                            value_match = re.search(r'₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)', value_text)
                            if value_match:
                                value = value_match.group(1).replace(',', '')
                                try:
                                    if '.' in value:
                                        value = float(value)
                                    else:
                                        value = int(value)
                                except ValueError:
                                    pass
                                
                                market_data[key] = value
    
    return market_data


def extract_years_data(section: Tag, metric_type: str) -> Dict[str, float]:
    """
    Extract data for different year periods (10 Years, 5 Years, 3 Years, TTM/Last Year)
    
    Args:
        section: BeautifulSoup Tag containing the growth metrics
        metric_type: Type of metric to extract
        
    Returns:
        Dictionary with year periods as keys and values as floats
    """
    years_data = {}
    
    # Look for year patterns in the section text
    section_text = section.text.lower()
    
    # Look for patterns like "10 Years: 21%"
    year_patterns = re.findall(r'(\d+)\s*years?[:\s]\s*(\d+(?:\.\d+)?)\s*%', section_text)
    ttm_pattern = re.findall(r'ttm[:\s]\s*(\d+(?:\.\d+)?)\s*%', section_text)
    last_year_pattern = re.findall(r'last\s+year[:\s]\s*(\d+(?:\.\d+)?)\s*%', section_text)
    
    # Process year patterns
    for years, value in year_patterns:
        years_data[f"{years} Years"] = float(value)
    
    # Process TTM
    if ttm_pattern:
        years_data["TTM"] = float(ttm_pattern[0])
    
    # Process Last Year
    if last_year_pattern:
        years_data["Last Year"] = float(last_year_pattern[0])
    
    return years_data


def extract_metrics_from_html(soup: BeautifulSoup) -> Dict[str, Dict[str, Any]]:
    """
    Extract metrics like growth rates and ratios from HTML content
    
    Args:
        soup: BeautifulSoup object containing the HTML content
        
    Returns:
        Dictionary mapping metric names to structured data
    """
    metrics_data = {
        "compounded_sales_growth": {},
        "compounded_profit_growth": {},
        "stock_price_cagr": {},
        "return_on_equity": {},
        "market_data": {}
    }
    
    # Extract market data
    market_data = extract_market_data(soup)
    if market_data:
        metrics_data["market_data"] = market_data
    
    # Define the metrics we're looking for with their possible HTML patterns
    target_metrics = {
        "compounded_sales_growth": [
            "Compounded Sales Growth", 
            "Sales Growth", 
            "Revenue Growth",
            "CAGR Sales"
        ],
        "compounded_profit_growth": [
            "Compounded Profit Growth", 
            "Profit Growth", 
            "Net Profit Growth",
            "CAGR Profit"
        ],
        "stock_price_cagr": [
            "Stock Price CAGR", 
            "Price CAGR", 
            "Share Price CAGR",
            "CAGR Stock"
        ],
        "return_on_equity": [
            "Return on Equity", 
            "ROE", 
            "Return on Shareholders' Equity",
            "Average ROE"
        ]
    }
    
    # Look for specific growth metric sections first
    growth_sections = soup.find_all(['div', 'section'], class_=lambda c: c and any(x in (c.lower() if c else '') for x in 
                                                                                 ['growth', 'metrics', 'cagr', 'compounded', 'return']))
    
    # Process each section that might contain growth metrics
    for section in growth_sections:
        # Check if this section has a heading that indicates it contains metrics
        heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5'])
        section_text = section.text.lower()
        
        # Check for each metric type
        for metric_key, terms in target_metrics.items():
            if any(term.lower() in section_text for term in terms):
                # Extract years data for this metric
                years_data = extract_years_data(section, metric_key)
                if years_data:
                    metrics_data[metric_key] = years_data
    
    # If we didn't find structured years data, try to extract individual values
    for metric_key, terms in target_metrics.items():
        if not metrics_data[metric_key]:
            # Look for any div that might contain this metric
            for div in soup.find_all('div'):
                div_text = div.text.lower()
                if any(term.lower() in div_text for term in terms):
                    # Try to extract years data
                    years_data = extract_years_data(div, metric_key)
                    if years_data:
                        metrics_data[metric_key] = years_data
                        break
    
    # If we still didn't find years data, look for individual metrics
    for metric_key, terms in target_metrics.items():
        if not metrics_data[metric_key]:
            for term in terms:
                # Search in table cells first (most reliable)
                for row in soup.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        if term.lower() in cell.text.lower():
                            # Check the next cell for the value
                            if i + 1 < len(cells):
                                value_cell = cells[i + 1]
                                value = value_cell.text.strip()
                                # Extract numeric value with % if present
                                match = re.search(r'[-+]?\d+\.?\d*\s*%?', value)
                                if match:
                                    value = match.group(0).strip()
                                    # Store as a single value
                                    metrics_data[metric_key] = {
                                        "10 Years": float(value.replace('%', ''))
                                    }
                                    break
                
                # If found, move to next metric
                if metrics_data[metric_key]:
                    break
                    
                # Search in other elements if not found in tables
                for elem in soup.find_all(['div', 'p', 'span', 'li']):
                    if term.lower() in elem.text.lower():
                        # Try to extract the value
                        text = elem.text.strip()
                        value_pattern = r'{}[\s:]+([-\d\.\+\-]+\s*%?)'.format(re.escape(term.lower()))
                        match = re.search(value_pattern, text.lower(), re.IGNORECASE)
                        
                        if match:
                            value = match.group(1).strip().replace('%', '')
                            try:
                                value_float = float(value)
                                # Store as a single value
                                metrics_data[metric_key] = {
                                    "10 Years": value_float
                                }
                                break
                            except ValueError:
                                pass
                
                # If found, move to next metric
                if metrics_data[metric_key]:
                    break
    
    return metrics_data


def extract_period_from_text(text: str) -> str:
    """
    Extract time period information from text
    
    Args:
        text: Text that might contain period information
        
    Returns:
        Extracted period or default value
    """
    # Common period patterns
    period_patterns = [
        # Years pattern (e.g., "10 Years" or "5-Year")
        r'(\d+)[-\s]?years?',
        r'(\d+)[-\s]?yr',
        # Specific date ranges
        r'(\d{4}[-/]\d{4})',  # e.g., 2019-2023
        r'(\w{3}\s+\d{4}[-/]\w{3}\s+\d{4})'  # e.g., Mar 2019-Mar 2023
    ]
    
    for pattern in period_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    # Default period if none found
    return "N/A"


def extract_sections_from_markdown(markdown_content: str) -> Dict[str, Any]:
    """
    Extract structured data from markdown content
    
    Args:
        markdown_content: Markdown content from crawl4ai
        
    Returns:
        Dictionary with extracted sections
    """
    sections = {
        "quarterly_results": {},
        "profit_loss": {},
        "compounded_sales_growth": {},
        "compounded_profit_growth": {},
        "stock_price_cagr": {},
        "return_on_equity": {},
        "balance_sheet": {},
        "cash_flows": {},
        "ratios": {},
        "shareholding_pattern": {}
    }
    
    # First, check if we have any tables in the markdown
    tables = extract_tables(markdown_content)
    
    # Process each table based on its heading
    for table_info in tables:
        heading = table_info.get('heading', '').strip()
        table_content = table_info.get('content', '')
        
        if 'Quarterly Results' in heading:
            sections["quarterly_results"] = parse_table(table_content)
        elif 'Profit & Loss' in heading:
            sections["profit_loss"] = parse_table(table_content)
        elif 'Balance Sheet' in heading:
            sections["balance_sheet"] = parse_table(table_content)
        elif 'Cash Flow' in heading:
            sections["cash_flows"] = parse_table(table_content)
        elif 'Ratios' in heading:
            sections["ratios"] = parse_table(table_content)
        elif 'Shareholding Pattern' in heading:
            sections["shareholding_pattern"] = parse_table(table_content)
    
    # Extract growth metrics and other key indicators
    # These are typically found in the 'Peer Comparison' section or standalone sections
    metrics_to_extract = [
        ("compounded_sales_growth", "Compounded Sales Growth"),
        ("compounded_profit_growth", "Compounded Profit Growth"),
        ("stock_price_cagr", "Stock Price CAGR"),
        ("return_on_equity", "Return on Equity")
    ]
    
    for section_key, metric_name in metrics_to_extract:
        metric_value = extract_metric(markdown_content, metric_name)
        if metric_value.get('value') != 'N/A':
            sections[section_key] = metric_value
    
    return sections


def extract_tables(markdown_content: str) -> List[Dict[str, str]]:
    """
    Extract tables and their headings from markdown content
    
    Args:
        markdown_content: Markdown content from crawl4ai
        
    Returns:
        List of dictionaries with table headings and content
    """
    tables = []
    
    # Split by headings (# or ##)
    heading_pattern = r'(#+\s+.*?)(?=#+\s+|$)'  # Match headings and content until next heading or end
    import re
    sections = re.findall(heading_pattern, markdown_content, re.DOTALL)
    
    for section in sections:
        lines = section.strip().split('\n')
        if not lines:  # Skip empty sections
            continue
            
        # Extract heading
        heading = lines[0].strip('#').strip()
        content = '\n'.join(lines[1:]).strip()
        
        # Check if this section contains a table
        if '|' in content and '---' in content:
            tables.append({
                'heading': heading,
                'content': content
            })
    
    # If no tables found with headings, try to find tables directly
    if not tables:
        # Find table patterns directly
        table_pattern = r'(\|.*\|\s*\n\|[-:\|\s]+\|\s*\n(?:\|.*\|\s*\n)+)'  # Match markdown tables
        table_matches = re.findall(table_pattern, markdown_content)
        
        for i, table_content in enumerate(table_matches):
            # Try to find a heading before this table
            heading = f"Table {i+1}"  # Default heading
            content_before_table = markdown_content.split(table_content)[0]
            heading_match = re.search(r'#+\s+(.*?)\s*$', content_before_table, re.MULTILINE)
            if heading_match:
                heading = heading_match.group(1).strip()
                
            tables.append({
                'heading': heading,
                'content': table_content
            })
    
    return tables


def parse_table(markdown_table: str) -> Dict[str, Any]:
    """
    Parse a markdown table into a structured dictionary
    
    Args:
        markdown_table: Markdown table content
        
    Returns:
        Dictionary representation of the table
    """
    result = {"headers": [], "rows": []}
    
    # Clean up the input
    lines = [line.strip() for line in markdown_table.strip().split("\n") if line.strip()]
    
    # Filter to only keep table lines (those starting with |)
    table_lines = [line for line in lines if line.startswith('|')]
    
    if len(table_lines) < 3:  # Need at least header, separator, and one data row
        return result
    
    # Extract headers (first line)
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.split("|")[1:-1]]
    result["headers"] = headers
    
    # Skip the separator line (second line)
    # Extract rows (remaining lines)
    rows = []
    for i in range(2, len(table_lines)):
        line = table_lines[i]
        row_values = [cell.strip() for cell in line.split("|")[1:-1]]
        
        # Make sure we have the right number of values
        if len(row_values) != len(headers):
            # Try to fix by padding or truncating
            if len(row_values) < len(headers):
                row_values.extend([''] * (len(headers) - len(row_values)))
            else:
                row_values = row_values[:len(headers)]
        
        row_dict = {headers[j]: row_values[j] for j in range(len(headers))}
        rows.append(row_dict)
    
    result["rows"] = rows
    return result


def extract_metric(content: str, metric_name: str) -> Dict[str, str]:
    """
    Extract a specific metric from the content
    
    Args:
        content: Section content
        metric_name: Name of the metric to extract
        
    Returns:
        Dictionary with the metric value
    """
    result = {"value": "N/A"}
    
    # Try different patterns to match the metric
    import re
    patterns = [
        # Pattern 1: Metric name followed by value
        rf'{re.escape(metric_name)}\s*[:\-]?\s*([-+]?\d+\.?\d*\s*%?)',
        # Pattern 2: Metric name in a table cell followed by value
        rf'\|\s*{re.escape(metric_name)}\s*\|\s*([-+]?\d+\.?\d*\s*%?)\s*\|',
        # Pattern 3: Metric as a key-value pair
        rf'["\']?{re.escape(metric_name)}["\']?\s*:\s*["\']?([-+]?\d+\.?\d*\s*%?)["\']?'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            result["value"] = match.group(1).strip()
            return result
    
    # If specific patterns don't match, try a more general approach
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if metric_name in line:
            # Try to extract the value from this line or the next
            current_line = line
            next_line = lines[i+1] if i+1 < len(lines) else ""
            
            # Look for numbers in current line after the metric name
            parts = current_line.split(metric_name)
            if len(parts) > 1:
                value_part = parts[1].strip()
                match = re.search(r'[-+]?\d+\.?\d*\s*%?', value_part)
                if match:
                    result["value"] = match.group(0).strip()
                    return result
            
            # If not found, check the next line for a number
            match = re.search(r'[-+]?\d+\.?\d*\s*%?', next_line)
            if match:
                result["value"] = match.group(0).strip()
                return result
    
    return result


async def process_stock_list(csv_file_path: str, output_dir: str = "./Historical_data/stock_data", limit: int = None) -> None:
    """
    Process all stocks in the CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing stock symbols
        output_dir: Directory to save the output JSON files
        limit: Optional limit on number of stocks to process (for testing)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read stock symbols from CSV
    symbols = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            symbols.append(row['Symbol'])
    
    # Apply limit if specified
    if limit and limit > 0:
        symbols = symbols[:limit]
        print(f"Limited to processing {limit} stocks out of {len(symbols)} total stocks")
    else:
        print(f"Found {len(symbols)} stocks to process")
    
    # Check which symbols already have data
    existing_symbols = set()
    for filename in os.listdir(output_dir):
        if filename.endswith(".json") and not filename.endswith("_error.json"):
            existing_symbols.add(filename.split(".")[0])
    
    # Filter out symbols that already have data
    symbols_to_process = [symbol for symbol in symbols if symbol not in existing_symbols]
    
    if len(symbols_to_process) < len(symbols):
        print(f"Skipping {len(symbols) - len(symbols_to_process)} symbols that already have data")
        print(f"Processing {len(symbols_to_process)} remaining symbols")
    
    # Process each stock
    async with AsyncWebCrawler() as crawler:
        for i, symbol in enumerate(symbols_to_process):

            try:
                # Add a random delay between requests to avoid rate limiting (10-20 seconds)
                if i > 0:
                    sleep_time = random.uniform(10, 20)
                    print(f"Waiting for {sleep_time:.2f} seconds before processing next symbol...")
                    await asyncio.sleep(sleep_time)
                
                # Scrape data for the stock
                stock_data = await scrape_stock_data(crawler, symbol)
                
                # Remove raw HTML/markdown from production data to save space
                if "raw_html" in stock_data:
                    del stock_data["raw_html"]
                if "raw_markdown" in stock_data:
                    del stock_data["raw_markdown"]
                if "raw_text" in stock_data:
                    del stock_data["raw_text"]
                
                # Save data to JSON file
                output_file = os.path.join(output_dir, f"{symbol}.json")
                with open(output_file, 'w') as f:
                    json.dump(stock_data, f, indent=2)
                
                print(f"Processed {i+1}/{len(symbols_to_process)}: {symbol}")
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                
                # Save error information
                error_file = os.path.join(output_dir, f"{symbol}_error.json")
                with open(error_file, 'w') as f:
                    json.dump({
                        "symbol": symbol,
                        "error": str(e),
                        "timestamp": time.time()
                    }, f, indent=2)
                
                # Continue with the next symbol
                continue


async def main():
    """Main function to run the scraper"""
    # Path to the CSV file
    csv_file_path = "/Users/abhishek/python_venv/fin_agent/Historical_data/ind_niftytotalmarket_list.csv"
    

    # Process all stocks in the CSV
    await process_stock_list(csv_file_path)


if __name__ == "__main__":
    asyncio.run(main())

