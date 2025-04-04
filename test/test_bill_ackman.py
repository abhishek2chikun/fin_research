#!/usr/bin/env python3
"""
Test script for the Bill Ackman agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from slave_agent.bill_ackman_agno import BillAckmanAgnoAgent
    
    # Create the agent
    agent = BillAckmanAgnoAgent()
    
    # Test state
    test_state = {
        "data": {
            "tickers": ["HDFCBANK"],
            "end_date": "2023-12-31"
        }
    }
    
    print("Testing Bill Ackman agent with state:", test_state)
    
    # Run the agent
    results = agent.run(test_state)
    
    # Print results
    print("\nResults:")
    for ticker, result in results.get("bill_ackman_agno_agent", {}).items():
        if "error" in result:
            print(f"Error for {ticker}: {result['error']}")
        else:
            print(f"Signal for {ticker}: {result.get('signal', 'unknown')} with confidence {result.get('confidence', 0)}")
            print(f"Reasoning: {result.get('reasoning', 'Not provided')[:100]}...")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are installed.")
except Exception as e:
    print(f"Error: {e}") 