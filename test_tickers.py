import yfinance as yf
import pandas as pd

# Try various gold-related tickers
tickers_to_try = [
    "GC=F",           # Gold futures (USD)
    "GOLD.BO",        # Gold on BSE India
    "MCXGOLD",        # MCX Gold
    "BUX.BO",         # Another India ticker
    "GOLDBEES.NS",    # Our current ticker (ETF)
]

for ticker in tickers_to_try:
    print(f"\n{'='*60}")
    print(f"Trying: {ticker}")
    print('='*60)
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if data is not None and not data.empty:
            close = data['Close'].iloc[-1]
            print(f"✓ SUCCESS")
            print(f"  Last Close: {close:.2f}")
            print(f"  Date: {data.index[-1]}")
            print(f"  Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
        else:
            print(f"✗ No data returned")
    except Exception as e:
        print(f"✗ Error: {str(e)[:120]}")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("Training data prices: ₹90,000+ (gold per 10g)")
print("Need to find ticker matching this scale")
