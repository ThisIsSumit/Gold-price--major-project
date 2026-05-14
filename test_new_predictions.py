import requests
import json

# Get fresh snapshot
r = requests.get('http://localhost:8001/api/v1/snapshot/live?force_retrain=true', timeout=30)
data = r.json()

latest_close = data.get('latest_close')
print(f'Latest close: ₹{latest_close:.2f}' if latest_close else 'Latest close: None')
print(f'Market status: {data.get("market_status")}')

preds = data.get('predictions', [])
print(f'\nPredictions: {len(preds)} articles\n')

if len(preds) > 0:
    print('First 3 predictions:')
    for i, p in enumerate(preds[:3]):
        pred_price = p["predicted_price"]
        pred_change = p["predicted_change_percent"]
        
        # Calculate expected price
        if latest_close:
            expected = latest_close * (1 + pred_change / 100)
            match = abs(pred_price - expected) < 1
        else:
            expected = None
            match = False
        
        print(f'{i+1}. Predicted: ₹{pred_price:.2f}')
        print(f'   Change: {pred_change:.4f}%')
        if expected:
            print(f'   Expected: ₹{latest_close:.2f} × (1 + {pred_change:.4f}%) = ₹{expected:.2f}')
            print(f'   Formula match: {match}')
        print()
    
    # Show price range
    prices = [p["predicted_price"] for p in preds]
    print(f'All predictions price range: ₹{min(prices):.2f} to ₹{max(prices):.2f}')
else:
    print('No predictions generated!')
