#!/usr/bin/env python3
"""
Test script for Ben Graham Agent

This script demonstrates how to use the Ben Graham agent
to analyze stocks based on Benjamin Graham's value investing principles.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent
try:
    from slave_agent.ben_graham_agno import BenGrahamAgnoAgent
except ImportError as e:
    print(f"Error importing Ben Graham Agent: {e}")
    print("Make sure you're running this script from the project root or test directory.")
    sys.exit(1)

def main():
    """Run Ben Graham agent test"""
    print("Testing Ben Graham Agent...")
    
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
    try:
        agent = BenGrahamAgnoAgent()
        print("Successfully initialized Ben Graham Agent.")
        
        # Run the agent analysis
        print("Running Ben Graham analysis...")
        results = agent.run(test_state)
        
        # Print summary of results
        print(f"\nAnalysis completed for {len(results.get('ben_graham_agent', {}))} tickers")
        
        # Print summary of signals
        for ticker, analysis in results.get('ben_graham_agent', {}).items():
            if "error" in analysis:
                print(f"Error for {ticker}: {analysis['error']}")
            else:
                signal = analysis.get("signal", "unknown")
                confidence = analysis.get("confidence", 0.0)
                score = analysis.get("analysis", {}).get("score", 0)
                max_score = analysis.get("analysis", {}).get("max_score", 0)
                print(f"{ticker}: Signal={signal}, Confidence={confidence:.2f}, Score={score}/{max_score}")
                
        print("\nBen Graham Agent test completed successfully.")
    except Exception as e:
        print(f"Error running Ben Graham Agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 