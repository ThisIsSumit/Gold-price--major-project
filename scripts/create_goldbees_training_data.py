"""
Create training data using GOLDBEES.NS (₹130 scale) instead of XAU_INR (₹90,000 scale).
This fixes the scale mismatch between training and live prediction data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from gold_app.data_sources import _download_ticker, _normalize_market_history_frame

def create_goldbees_training_data():
    """Download 1 year of GOLDBEES.NS data and save as training market file."""
    print("📥 Downloading GOLDBEES.NS historical data (1 year)...")
    
    # Download GOLDBEES data
    frame = _download_ticker("GOLDBEES.NS", period="1y")
    frame = _normalize_market_history_frame(frame)
    
    if frame.empty:
        print("❌ Failed to fetch GOLDBEES.NS data")
        return False
    
    print(f"✓ Downloaded {len(frame)} rows of GOLDBEES.NS data")
    print(f"  Date range: {frame['Date'].min()} to {frame['Date'].max()}")
    print(f"  Price range: ₹{frame['Close'].min():.2f} to ₹{frame['Close'].max():.2f}")
    
    # Convert to the format expected by training pipeline
    # XAU_INR format: Date, Price, Open, High, Low, Vol., Change %
    training_frame = pd.DataFrame({
        'Date': pd.to_datetime(frame['Date']).dt.strftime('%m/%d/%Y'),
        'Price': frame['Close'],
        'Open': frame['Open'],
        'High': frame['High'],
        'Low': frame['Low'],
        'Vol.': frame['Volume'].fillna(0),
    })

    # Use real close-to-close changes so the model learns a non-zero target.
    training_frame['Change %'] = training_frame['Price'].pct_change().mul(100.0).fillna(0.0).round(4)
    
    # Save to same location as XAU_INR data
    output_path = Path(__file__).parent.parent / "GOLDBEES_NS_training_data.csv"
    training_frame.to_csv(output_path, index=False)
    
    print(f"\n✅ Created training data file: {output_path}")
    print(f"   Rows: {len(training_frame)}")
    print(f"   Columns: {list(training_frame.columns)}")
    print("\n📝 Next step: Update config.py to use this file:")
    print("   TRAINING_MARKET_PATH = ROOT_DIR / 'GOLDBEES_NS_training_data.csv'")
    
    return True

if __name__ == "__main__":
    success = create_goldbees_training_data()
    sys.exit(0 if success else 1)
