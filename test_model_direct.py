import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from gold_app.model import load_model
from gold_app.data_sources import fetch_market_data
import pandas as pd

print('Loading model...')
model = load_model()
print(f'Model version: {model.version}')
print(f'Model metrics: {model.metrics}')

print('\nFetching live market data...')
frame = fetch_market_data()
print(f'Live data shape: {frame.shape}')
print(f'Latest close: ₹{frame["Close"].iloc[-1]:.2f}' if len(frame) > 0 else 'No data')

# Get the latest market features
if len(frame) > 1:
    latest_close = frame['Close'].iloc[-1]
    prev_close = frame['Close'].iloc[-2]
    
    # Show what the model sees
    print(f'\nLive data features for latest close (₹{latest_close:.2f}):')
    print(f'  Close: {latest_close}')
    print(f'  Prev_Close: {prev_close}')
    print(f'  Open: {frame["Open"].iloc[-1]}')
    
    # Try a prediction on sample news
    from sklearn.compose import ColumnTransformer
    
    # Use pipeline to transform
    sample_news = "Gold prices surge as market volatility increases"
    sample_features = {
        'News': sample_news,
        'Open': frame['Open'].iloc[-1],
        'Prev_Close': prev_close,
        'Prev_Open': frame['Open'].iloc[-2] if len(frame) > 2 else frame['Open'].iloc[-1],
        'Prev_Change': 0,
        'MA_3': frame['Close'].iloc[-3:].mean(),
        'MA_7': frame['Close'].iloc[-7:].mean() if len(frame) > 7 else frame['Close'].mean(),
        'STD_3': frame['Close'].iloc[-3:].std(),
        'Year': pd.Timestamp.now().year,
        'Month': pd.Timestamp.now().month,
        'Day': pd.Timestamp.now().day,
    }
    
    # Create test dataframe
    test_df = pd.DataFrame([sample_features])
    
    print(f'\nTest prediction on: "{sample_news}"')
    print(f'Test features: {test_df.to_dict()}')
    
    try:
        prediction = model.regressor.predict(test_df[['News'] + model.feature_columns])
        print(f'Prediction output shape: {prediction.shape}')
        print(f'Predicted price: ₹{prediction[0][0]:.2f}')
        print(f'Predicted change: {prediction[0][1]:.4f}%')
    except Exception as e:
        print(f'Prediction error: {e}')
else:
    print('Insufficient data')
