import pandas as pd
try:
    import yfinance as yf
except:
    print("ERROR: yfinance not installed")
    exit(1)

print("Testing GC=F...")
data = yf.download("GC=F", period="1y", interval="1d", auto_adjust=False, progress=False, threads=False)
frame = data.reset_index()
print(f"MultiIndex: {isinstance(frame.columns, pd.MultiIndex)}")
if isinstance(frame.columns, pd.MultiIndex):
    frame.columns = [col[0] if isinstance(col, tuple) else str(col) for col in frame.columns]
print(f"Columns: {list(frame.columns)}")
print(f"Has Date: {'Date' in frame.columns}, Has Close: {'Close' in frame.columns}")
print(f"First Close: {frame['Close'].iloc[0]:.2f}")

print("\nTesting USD/INR rate...")
rate_data = yf.download("USDINR=X", period="1d", interval="1d", auto_adjust=False, progress=False, threads=False)
rate = float(rate_data["Close"].iloc[-1])
print(f"Rate: {rate:.2f}")
print(f"USD 3200 = INR {3200 * rate:.2f}")
