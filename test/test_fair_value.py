#!/usr/bin/env python3

from tools.fundamental import get_free_cash_flow, get_market_cap

def main():
    ticker = "HDFCBANK"
    
    # Get FCF data
    fcf_data = get_free_cash_flow(ticker)
    print(f"FCF data for {ticker}:", fcf_data)
    
    # Get market cap
    market_cap = get_market_cap(ticker)
    print(f"Market cap for {ticker}:", market_cap)
    
    # Calculate FCF yield if possible
    if fcf_data and "current_fcf" in fcf_data and fcf_data["current_fcf"] is not None and market_cap:
        current_fcf = fcf_data["current_fcf"]
        fcf_yield = current_fcf / market_cap
        print(f"FCF yield: {fcf_yield:.4f} ({fcf_yield*100:.2f}%)")
        
        # Check if FCF yield meets the threshold for fair value estimation
        if fcf_yield > 0.05:  # 5% threshold
            fair_value = current_fcf / 0.05
            print(f"Fair value estimate: {fair_value:,.2f} (Current market cap: {market_cap:,.2f})")
            print(f"Potential upside: {((fair_value / market_cap) - 1) * 100:.2f}%")
        else:
            print(f"FCF yield of {fcf_yield*100:.2f}% is below the 5% threshold for fair value estimation")
            print("This is why fair_value_estimate is null in the results")
    else:
        print("Cannot calculate FCF yield due to missing data")

if __name__ == "__main__":
    main() 