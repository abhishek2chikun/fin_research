#!/usr/bin/env python3
"""
Test script for Ben Graham Agno agent

This script demonstrates how to use the Ben Graham Agno agent
to analyze stocks based on Benjamin Graham's value investing principles.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent
from slave_agent.ben_graham_agno import BenGrahamAgnoAgent

def main():
    """Run Ben Graham Agno agent test"""
    print("Testing Ben Graham Agno Agent...")
    
    # Create test state with tickers to analyze
    test_state = {
        "data": {
            "tickers": ["HDFCBANK", "RELIANCE"],  # Example tickers
            "end_date": datetime.now().strftime("%Y-%m-%d")
        },
        "metadata": {
            "model_name": "sufe-aiflm-lab_fin-r1",
            "model_provider": "lmstudio"
        }
    }
    
    # Initialize and run the agent
    agent = BenGrahamAgnoAgent()
    try:
        results = agent.run(test_state)
        print(json.dumps(results, indent=2))
        print("\nBen Graham Agno Agent test completed successfully.")
    except Exception as e:
        print(f"Error running Ben Graham Agno Agent: {e}")

if __name__ == "__main__":
    main() 