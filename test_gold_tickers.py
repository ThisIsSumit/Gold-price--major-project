import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# List of tickers to test
tickers = ['GC=F', 'GLD', 'IAU', 'BZ=F', 'YG=F', 'GOLDGULD']

# Define date range - 1 year
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print("=" * 80)
print("GOLD PRICE TICKER TEST")
print("=" * 80)
print(f"Testing period: {start_date.date()} to {end_date.date()}")
print("=" * 80)

results = {'working': [], 'failed': []}

for ticker in tickers:
    print(f"\nTesting ticker: {ticker}")
    print("-" * 40)
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Check if we got data
        if data.empty:
            print(f"  FAILED: No data returned")
            results['failed'].append(ticker)
        else:
            print(f"  SUCCESS: Downloaded {len(data)} rows")
            print(f"\n  First few rows:")
            print(data.head(3).to_string())
            
            # Get the last close price
            try:
                last_close = float(data['Close'].iloc[-1])
                last_date = data.index[-1]
                print(f"\n  Last Close Price: {last_close:.2f}")
                print(f"  Date of last price: {last_date.date()}")
            except:
                print(f"\n  Last row: {data.iloc[-1].to_string()}")
            
            results['working'].append(ticker)
            
    except Exception as e:
        err_str = str(e)[:100]
        print(f"  FAILED with error: {err_str}")
        results['failed'].append(ticker)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nWorking Tickers: {len(results['working'])}")
for ticker in results['working']:
    print(f"  - {ticker}")

print(f"\nFailed Tickers: {len(results['failed'])}")
for ticker in results['failed']:
    print(f"  - {ticker}")
print("=" * 80)
