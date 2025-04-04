#!/usr/bin/env python3
"""
Tools for fetching and processing OHLCV (Open, High, Low, Close, Volume) data.

These tools allow the financial agents to retrieve, analyze, and process
historical price data for a given ticker symbol.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel

# Define OHLCVBar and OHLCVData classes here to avoid data_model dependency
class OHLCVBar(BaseModel):
    """Model representing a single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bar to a dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }

class OHLCVData(BaseModel):
    """Model representing a collection of OHLCV bars for a ticker."""
    ticker: str
    timeframe: str
    bars: List[OHLCVBar]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the OHLCV data to pandas DataFrame."""
        data = []
        for bar in self.bars:
            data.append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the OHLCV data to a dictionary."""
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "bars": [bar.to_dict() for bar in self.bars]
        }
    
    @classmethod
    def from_csv(cls, file_path: str, ticker: str, timeframe: str = '1d') -> 'OHLCVData':
        """
        Create an OHLCVData object from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            ticker: Ticker symbol for this data
            timeframe: Timeframe of the data (e.g., '1d', '1h', '5m')
            
        Returns:
            OHLCVData object
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            date_col = None
            
            # Check for date/time column
            for col in ['datetime', 'date', 'timestamp']:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                raise ValueError("CSV file must have a datetime, date, or timestamp column")
            
            # Convert date column to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Handle optional columns
            for col in ['stock_code', 'open_interest']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Ensure all required columns are present and numeric
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create OHLCVBar objects
            bars = []
            for _, row in df.iterrows():
                bars.append(OHLCVBar(
                    timestamp=row[date_col],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                ))
            
            # Create and return OHLCVData object
            return cls(ticker=ticker, timeframe=timeframe, bars=bars)
        
        except Exception as e:
            raise Exception(f"Error creating OHLCVData from CSV: {e}")

# Base directory for historical OHLCV data
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Historical_data', 'ohlcv')

def fetch_ohlcv_data(ticker: str, timeframe: str = '1d') -> Optional[OHLCVData]:
    """
    Fetch OHLCV data for a given ticker symbol from the local storage.
    
    Args:
        ticker: The ticker symbol to fetch data for
        timeframe: Timeframe of the data (e.g., '1d', '1h', '5m')
        
    Returns:
        OHLCVData object if found, None otherwise
    """
    ticker = ticker.upper()  # Ensure ticker is in uppercase
    
    # Construct path to the ticker's OHLCV data file
    file_path = os.path.join(BASE_DIR, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"OHLCV data not found for {ticker} at {file_path}")
        return None
    
    try:
        # Use the OHLCVData.from_csv method
        ohlcv_data = OHLCVData.from_csv(file_path, ticker, timeframe)
        return ohlcv_data
    except Exception as e:
        print(f"Error loading OHLCV data for {ticker}: {e}")
        return None

def get_ohlcv_data(ticker: str, timeframe: str = '1d') -> Optional[OHLCVData]:
    """
    Alias for fetch_ohlcv_data for compatibility.
    
    Args:
        ticker: The ticker symbol to fetch data for
        timeframe: The timeframe for the data (e.g., '1d', '1h')
        
    Returns:
        OHLCVData object or None if data not found
    """
    return fetch_ohlcv_data(ticker, timeframe)

def get_price_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Get price data for a given ticker within a specified date range.
    
    Args:
        ticker: The ticker symbol
        start_date: Start date in format 'YYYY-MM-DD' (optional)
        end_date: End date in format 'YYYY-MM-DD' (optional)
        
    Returns:
        Dictionary containing price data
    """
    ohlcv_data = fetch_ohlcv_data(ticker)
    if not ohlcv_data:
        return {}
    
    # Convert to DataFrame for easier date filtering
    df = ohlcv_data.to_dataframe()
    
    # Apply date filtering if provided
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        except Exception as e:
            print(f"Error parsing start_date {start_date}: {e}")
    
    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
        except Exception as e:
            print(f"Error parsing end_date {end_date}: {e}")
    
    # Convert filtered DataFrame back to dictionary format
    if df.empty:
        return {}
    
    # Get the latest price for quick access
    latest_price = df['close'].iloc[-1] if not df.empty else None
    
    # Calculate some basic statistics
    price_data = {
        "ticker": ticker,
        "latest_price": latest_price,
        "latest_date": df.index[-1].strftime("%Y-%m-%d") if not df.empty else None,
        "price_data": df.reset_index().to_dict(orient='records'),
        "summary": {
            "avg_price": df['close'].mean() if not df.empty else None,
            "min_price": df['close'].min() if not df.empty else None,
            "max_price": df['close'].max() if not df.empty else None,
            "std_dev": df['close'].std() if not df.empty else None,
            "period_return": ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) if len(df) > 1 else 0
        }
    }
    
    return price_data

def calculate_technical_indicators(ticker: str, period: int = 14) -> Dict[str, Any]:
    """
    Calculate various technical indicators for a given ticker.
    
    Args:
        ticker: The ticker symbol
        period: Period for technical indicators (default: 14 days)
        
    Returns:
        Dictionary containing technical indicators
    """
    ohlcv_data = fetch_ohlcv_data(ticker)
    if not ohlcv_data:
        # Return default structure with null values when data is not available
        return {
            "ticker": ticker,
            "latest_price": None,
            "latest_date": None,
            "trend": "unknown",
            "moving_averages": {
                "sma_50": None,
                "sma_200": None,
                "ema_20": None,
                "price_vs_sma50": None,
                "price_vs_sma200": None,
                "golden_cross": False,
                "death_cross": False
            },
            "oscillators": {
                "rsi": None,
                "rsi_status": "unknown"
            },
            "bollinger_bands": {
                "upper": None,
                "middle": None,
                "lower": None,
                "percent_b": None
            },
            "macd": {
                "macd_line": None,
                "signal_line": None,
                "histogram": None,
                "signal": "unknown"
            },
            "volume": {
                "latest": None,
                "avg_20d": None,
                "trend": "unknown"
            }
        }
    
    # Convert to DataFrame for calculations
    df = ohlcv_data.to_dataframe()
    if df.empty:
        return {}
    
    # Calculate moving averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    
    # Get the latest values for each indicator
    latest = df.iloc[-1]
    
    # Determine trend based on moving averages
    price = latest['close']
    trend = "neutral"
    if price > latest['sma_50'] and latest['sma_50'] > latest['sma_200']:
        trend = "bullish"
    elif price < latest['sma_50'] and latest['sma_50'] < latest['sma_200']:
        trend = "bearish"
    
    # Golden/Death Cross detection
    golden_cross = False
    death_cross = False
    if len(df) > 50:  # Need enough data for moving averages
        prev_50 = df['sma_50'].iloc[-2]
        prev_200 = df['sma_200'].iloc[-2]
        curr_50 = latest['sma_50']
        curr_200 = latest['sma_200']
        
        golden_cross = prev_50 <= prev_200 and curr_50 > curr_200
        death_cross = prev_50 >= prev_200 and curr_50 < curr_200
    
    # Compile all indicators
    indicators = {
        "ticker": ticker,
        "latest_price": price,
        "latest_date": df.index[-1].strftime("%Y-%m-%d"),
        "trend": trend,
        "moving_averages": {
            "sma_50": latest['sma_50'],
            "sma_200": latest['sma_200'],
            "ema_20": latest['ema_20'],
            "price_vs_sma50": (price / latest['sma_50'] - 1) * 100,  # % deviation
            "price_vs_sma200": (price / latest['sma_200'] - 1) * 100,  # % deviation
            "golden_cross": golden_cross,
            "death_cross": death_cross
        },
        "oscillators": {
            "rsi": latest['rsi'],
            "rsi_status": "oversold" if latest['rsi'] < 30 else "overbought" if latest['rsi'] > 70 else "neutral"
        },
        "bollinger_bands": {
            "upper": latest['upper_band'],
            "middle": latest['middle_band'],
            "lower": latest['lower_band'],
            "percent_b": (price - latest['lower_band']) / (latest['upper_band'] - latest['lower_band']) if (latest['upper_band'] - latest['lower_band']) != 0 else 0.5
        },
        "macd": {
            "macd_line": latest['macd_line'],
            "signal_line": latest['signal_line'],
            "histogram": latest['macd_histogram'],
            "signal": "bullish" if latest['macd_line'] > latest['signal_line'] else "bearish"
        }
    }
    
    return indicators

def convert_df_to_price_data(df, ticker):
    """
    Convert DataFrame to a structure that can be used by the analysis functions.
    This mimics the expected structure for price data.
    
    Args:
        df: Pandas DataFrame with OHLCV data
        ticker: Ticker symbol
        
    Returns:
        Object with attributes that mimic OHLCVData structure
    """
    if df is None or df.empty:
        return None
        
    # Create a structure with lists of data that the analysis functions expect
    return type('PriceData', (), {
        'ticker': ticker,
        'dates': df.index.tolist(),
        'opens': df['open'].tolist() if 'open' in df.columns else [],
        'highs': df['high'].tolist() if 'high' in df.columns else [],
        'lows': df['low'].tolist() if 'low' in df.columns else [],
        'closes': df['close'].tolist() if 'close' in df.columns else [],
        'volumes': df['volume'].tolist() if 'volume' in df.columns else []
    })

def load_filtered_price_data(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load price data from CSV file and filter by date range.
    
    Args:
        ticker: The ticker symbol
        start_date: Optional start date in 'YYYY-MM-DD' format
        end_date: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        Pandas DataFrame with filtered price data or None if data not available
    """
    # Construct path to the ticker's OHLCV data file
    file_path = os.path.join(BASE_DIR, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        print(f"OHLCV data not found for {ticker} at {file_path}")
        return None
    
    try:
        # Read CSV file directly
        df = pd.read_csv(file_path)
        
        # Convert datetime column to datetime and set as index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Drop unnecessary columns
        if 'stock_code' in df.columns:
            df.drop(columns=['stock_code'], inplace=True)
        if 'open_interest' in df.columns:
            df.drop(columns=['open_interest'], inplace=True)
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Filter by date if needed
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
            except Exception as e:
                print(f"Error parsing start_date {start_date}: {e}")
        
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            except Exception as e:
                print(f"Error parsing end_date {end_date}: {e}")
        
        if df.empty:
            print(f"No price data available for {ticker} in specified date range")
            return None
        
        # Print sample for debugging
        print(f"Loaded data for {ticker}. Sample:")
        print(df.head(3))
        print(f"Shape: {df.shape}, Columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"Error loading CSV data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

# List available tickers for which we have OHLCV data
def list_available_tickers() -> List[str]:
    """
    List all available tickers for which we have OHLCV data.
    
    Returns:
        List of ticker symbols
    """
    if not os.path.exists(BASE_DIR):
        return []
    
    csv_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.csv')]
    return [os.path.splitext(f)[0] for f in csv_files]
