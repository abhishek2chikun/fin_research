#!/usr/bin/env python3
"""
Test script for accessing OHLCV data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_model.ohlcv import OHLCVData, FinancialDataService
    from tools.ohlcv import fetch_ohlcv_data, get_price_data
    
    print("Testing OHLCV data access:")
    
    # Get a list of available tickers
    available_tickers = FinancialDataService.list_available_ohlcv_symbols()
    print(f"Found {len(available_tickers)} tickers with OHLCV data")
    print(f"First 5 tickers: {available_tickers[:5] if len(available_tickers) >= 5 else available_tickers}")
    
    # Select a test ticker
    test_ticker = available_tickers[0] if available_tickers else None
    
    if test_ticker:
        print(f"\nTesting with ticker: {test_ticker}")
        
        # Test direct access through FinancialDataService
        print("\nTest 1: Load through FinancialDataService")
        ohlcv_data = FinancialDataService.load_ohlcv_data(test_ticker)
        if ohlcv_data:
            print(f"Successfully loaded {len(ohlcv_data.bars)} OHLCV bars")
            print(f"First bar date: {ohlcv_data.bars[0].date if ohlcv_data.bars else 'N/A'}")
            print(f"Last bar date: {ohlcv_data.bars[-1].date if ohlcv_data.bars else 'N/A'}")
        else:
            print("Failed to load OHLCV data")
        
        # Test through tools.ohlcv
        print("\nTest 2: Load through fetch_ohlcv_data")
        ohlcv_data2 = fetch_ohlcv_data(test_ticker)
        if ohlcv_data2:
            print(f"Successfully loaded {len(ohlcv_data2.bars)} OHLCV bars")
        else:
            print("Failed to load OHLCV data through fetch_ohlcv_data")
        
        # Test get_price_data
        print("\nTest 3: Load through get_price_data")
        price_data = get_price_data(test_ticker)
        if price_data:
            print(f"Successfully loaded price data")
            print(f"Latest price: {price_data.get('latest_price')}")
            print(f"Period return: {price_data.get('summary', {}).get('period_return')}")
        else:
            print("Failed to load price data through get_price_data")
    else:
        print("No tickers found to test with")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}") 