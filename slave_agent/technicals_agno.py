"""
Technical Analysis Agent using Agno Framework
"""

from typing import Dict, List, Any, Optional
import json
import math
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing_extensions import Literal

# Import Agno framework
from agno.agent import Agent

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tools
from tools.ohlcv import fetch_ohlcv_data, OHLCVData, load_filtered_price_data, convert_df_to_price_data

# Pydantic model for the output signal
class TechnicalsSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: Dict[str, Any]


def calculate_trend_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Advanced trend following strategy using multiple timeframes and indicators
    """
    # Calculate EMAs for multiple timeframes
    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)

    # Calculate ADX for trend strength
    adx = calculate_adx(prices_df, 14)

    # Determine trend direction and strength
    short_trend = ema_8 > ema_21
    medium_trend = ema_21 > ema_55

    # Combine signals with confidence weighting
    trend_strength = adx["adx"].iloc[-1] / 100.0

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = "bullish"
        confidence = trend_strength
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "adx": float(adx["adx"].iloc[-1]),
            "trend_strength": float(trend_strength),
        },
    }


def calculate_mean_reversion_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mean reversion strategy using statistical measures and Bollinger Bands
    """
    # Calculate z-score of price relative to moving average
    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()
    z_score = (prices_df["close"] - ma_50) / std_50

    # Calculate Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)

    # Calculate RSI with multiple timeframes
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    # Mean reversion signals
    price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Combine signals
    if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
        signal = "bullish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
        signal = "bearish"
        confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "z_score": float(z_score.iloc[-1]),
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    }


def calculate_momentum_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Multi-factor momentum strategy
    """
    # Price momentum
    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    volume_momentum = prices_df["volume"] / volume_ma

    # Calculate momentum score
    momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]

    # Volume confirmation
    volume_confirmation = volume_momentum.iloc[-1] > 1.0

    if momentum_score > 0.05 and volume_confirmation:
        signal = "bullish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal = "bearish"
        confidence = min(abs(momentum_score) * 5, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]),
        },
    }


def calculate_volatility_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Volatility-based trading strategy
    """
    # Calculate various volatility metrics
    returns = prices_df["close"].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # ATR ratio
    atr = calculate_atr(prices_df)
    atr_ratio = atr / prices_df["close"]

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < 0.8 and vol_z < -1:
        signal = "bullish"  # Low vol regime, potential for expansion
        confidence = min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal = "bearish"  # High vol regime, potential for contraction
        confidence = min(abs(vol_z) / 3, 1.0)
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": float(current_vol_regime),
            "volatility_z_score": float(vol_z),
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    }


def calculate_stat_arb_signals(prices_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistical arbitrage signals based on price action analysis
    """
    # Calculate price distribution statistics
    returns = prices_df["close"].pct_change()

    # Skewness and kurtosis
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # Test for mean reversion using Hurst exponent
    hurst = calculate_hurst_exponent(prices_df["close"])

    # Generate signal based on statistical properties
    if hurst < 0.4 and skew.iloc[-1] > 1:
        signal = "bullish"
        confidence = (0.5 - hurst) * 2
    elif hurst < 0.4 and skew.iloc[-1] < -1:
        signal = "bearish"
        confidence = (0.5 - hurst) * 2
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": float(hurst),
            "skewness": float(skew.iloc[-1]),
            "kurtosis": float(kurt.iloc[-1]),
        },
    }


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent to determine long-term memory of time series
    H < 0.5: Mean reverting series
    H = 0.5: Random walk
    H > 0.5: Trending series

    Args:
        price_series: Array-like price data
        max_lag: Maximum lag for R/S calculation

    Returns:
        float: Hurst exponent
    """
    try:
        # Ensure we have enough data
        if len(price_series) < max_lag * 2:
            return 0.5  # Return random walk as default
        
        # Handle NaN values
        price_series = price_series.ffill().bfill()
        
        lags = range(2, max_lag)
        # Add small epsilon to avoid log(0)
        tau = []
        for lag in lags:
            # Calculate standard deviation of lagged differences
            diff = np.subtract(price_series[lag:].values, price_series[:-lag].values)
            std_dev = np.std(diff)
            if std_dev == 0:
                continue  # Skip this lag if standard deviation is zero
            tau.append(max(1e-8, std_dev))
        
        # If no valid lags, return random walk
        if not tau or len(tau) < 2:
            return 0.5
        
        # Return the Hurst exponent from linear fit
        reg = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        hurst = reg[0]
        
        # Sanity check on result
        if not np.isfinite(hurst) or hurst < 0 or hurst > 1:
            return 0.5  # Return random walk as default
            
        return hurst
        
    except Exception as e:
        print(f"Error calculating Hurst exponent: {e}")
        # Return 0.5 (random walk) if calculation fails
        return 0.5


def weighted_signal_combination(signals: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Combines multiple trading signals using a weighted approach
    """
    # Convert signals to numeric values
    signal_values = {"bullish": 1, "neutral": 0, "bearish": -1}

    weighted_sum = 0
    total_confidence = 0

    for strategy, signal in signals.items():
        try:
            numeric_signal = signal_values[signal["signal"]]
            weight = weights.get(strategy, 0)
            confidence = min(max(signal.get("confidence", 0.5), 0), 1)  # Ensure confidence is between 0 and 1

            weighted_sum += numeric_signal * weight * confidence
            total_confidence += weight * confidence
        except (KeyError, TypeError) as e:
            print(f"Error processing signal for {strategy}: {e}")
            continue

    # Normalize the weighted sum
    if total_confidence > 0:
        final_score = weighted_sum / total_confidence
    else:
        final_score = 0

    # Convert back to signal
    if final_score > 0.2:
        signal = "bullish"
    elif final_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    # Ensure confidence is a valid value
    final_confidence = abs(final_score)
    if not np.isfinite(final_confidence):
        final_confidence = 0.5
    
    return {"signal": signal, "confidence": min(final_confidence, 1.0)}


# Technical indicator calculation functions
def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    try:
        delta = prices_df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to neutral RSI when calculation fails
    except Exception as e:
        print(f"Error calculating RSI: {e}")
        # Return a Series filled with 50 (neutral) values
        return pd.Series(50, index=prices_df.index)


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = prices_df["close"].rolling(window).mean()
    std_dev = prices_df["close"].rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        df: DataFrame with price data
        window: EMA period

    Returns:
        pd.Series: EMA values
    """
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with OHLC data
        period: Period for calculations

    Returns:
        DataFrame with ADX values
    """
    # Calculate True Range
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # Calculate Directional Movement
    df["up_move"] = df["high"] - df["high"].shift()
    df["down_move"] = df["low"].shift() - df["low"]

    df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
    df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)

    # Calculate ADX
    df["+di"] = 100 * (df["plus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["-di"] = 100 * (df["minus_dm"].ewm(span=period).mean() / df["tr"].ewm(span=period).mean())
    df["dx"] = 100 * abs(df["+di"] - df["-di"]) / (df["+di"] + df["-di"])
    df["adx"] = df["dx"].ewm(span=period).mean()

    return df[["adx", "+di", "-di"]]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range

    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        pd.Series: ATR values
    """
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    return true_range.rolling(period).mean()


def normalize_pandas(obj):
    """Convert pandas Series/DataFrames to primitive Python types"""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(item) for item in obj]
    return obj


def transform_prices_to_df(prices_data: OHLCVData) -> pd.DataFrame:
    """
    Transform OHLCV data from the API into a pandas DataFrame.
    """
    if not prices_data or not hasattr(prices_data, 'bars') or not prices_data.bars:
        return None
    
    # Use the built-in to_dataframe method of OHLCVData
    return prices_data.to_dataframe()


class TechnicalsAgnoAgent():
    """Agno-based agent implementing technical analysis."""
    
    def __init__(self):
        pass
        
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes stocks using technical analysis methodologies.
        """
        agent_name = "technicals_agno_agent"
        
        data = state.get("data", {})
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        tickers = data.get("tickers", [])
        
        if not tickers:
            return {f"{agent_name}_error": "Missing 'tickers' in input state."}
        
        results = {}
        
        for ticker in tickers:
            try:
                print(f"Analyzing {ticker} with technical analysis...")
                
                # Get historical price data
                
                # First try to get filtered data as DataFrame
                prices_df = load_filtered_price_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if prices_df is None:
                    # Fallback to unfiltered data if filtered data isn't available
                    ohlcv_data = fetch_ohlcv_data(ticker)
                    if not ohlcv_data:
                        results[ticker] = {"error": f"No price data found for {ticker}"}
                        continue
                    
                    # Convert to DataFrame using the OHLCVData's built-in method
                    prices_df = ohlcv_data.to_dataframe()
                    
                    # Apply date filtering manually if needed
                    if start_date:
                        try:
                            start_dt = pd.to_datetime(start_date)
                            prices_df = prices_df[prices_df.index >= start_dt]
                        except Exception as e:
                            print(f"Error parsing start_date {start_date}: {e}")
                    
                    if end_date:
                        try:
                            end_dt = pd.to_datetime(end_date)
                            prices_df = prices_df[prices_df.index <= end_dt]
                        except Exception as e:
                            print(f"Error parsing end_date {end_date}: {e}")
                
                if prices_df is None or prices_df.empty:
                    results[ticker] = {"error": f"No valid price data found for {ticker} in specified date range"}
                    continue
                
                # Check if we have enough data points for technical analysis
                min_required_bars = 200  # Require at least 200 bars for reliable indicators
                if len(prices_df) < min_required_bars:
                    results[ticker] = {
                        "error": f"Insufficient price data for {ticker}. Technical analysis requires at least {min_required_bars} bars, but only {len(prices_df)} available."
                    }
                    continue
                
                # Calculate technical signals
                try:
                    trend_signals = calculate_trend_signals(prices_df)
                    mean_reversion_signals = calculate_mean_reversion_signals(prices_df)
                    momentum_signals = calculate_momentum_signals(prices_df)
                    volatility_signals = calculate_volatility_signals(prices_df)
                    stat_arb_signals = calculate_stat_arb_signals(prices_df)
                except Exception as e:
                    print(f"Error calculating technical signals for {ticker}: {e}")
                    import traceback
                    traceback.print_exc()
                    results[ticker] = {"error": f"Error calculating technical signals: {str(e)}"}
                    continue
                
                # Combine all signals using a weighted ensemble approach
                strategy_weights = {
                    "trend": 0.25,
                    "mean_reversion": 0.20,
                    "momentum": 0.25,
                    "volatility": 0.15,
                    "stat_arb": 0.15,
                }
                
                combined_signal = weighted_signal_combination(
                    {
                        "trend": trend_signals,
                        "mean_reversion": mean_reversion_signals,
                        "momentum": momentum_signals,
                        "volatility": volatility_signals,
                        "stat_arb": stat_arb_signals,
                    },
                    strategy_weights,
                )
                
                # Store results
                results[ticker] = {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": {
                        "trend_following": {
                            "signal": trend_signals["signal"],
                            "confidence": round(trend_signals["confidence"] * 100),
                            "metrics": normalize_pandas(trend_signals["metrics"]),
                        },
                        "mean_reversion": {
                            "signal": mean_reversion_signals["signal"],
                            "confidence": round(mean_reversion_signals["confidence"] * 100),
                            "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                        },
                        "momentum": {
                            "signal": momentum_signals["signal"],
                            "confidence": round(momentum_signals["confidence"] * 100),
                            "metrics": normalize_pandas(momentum_signals["metrics"]),
                        },
                        "volatility": {
                            "signal": volatility_signals["signal"],
                            "confidence": round(volatility_signals["confidence"] * 100),
                            "metrics": normalize_pandas(volatility_signals["metrics"]),
                        },
                        "statistical_arbitrage": {
                            "signal": stat_arb_signals["signal"],
                            "confidence": round(stat_arb_signals["confidence"] * 100),
                            "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                        },
                    },
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[ticker] = {"error": f"Error analyzing {ticker}: {str(e)}"}
        
        return {agent_name: results}


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_state = {
        "data": {
            "tickers": ["AAPL"],  # Example ticker
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
    }
    try:
        agent = TechnicalsAgnoAgent()
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error running example: {e}")
        print("Ensure price data is properly set up.") 