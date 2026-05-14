import requests
import json

# Get latest snapshot
r = requests.get('http://localhost:8001/api/v1/snapshot/latest', timeout=10)
data = r.json()

print('Market status:', data.get('market_status'))
print('Latest close:', data.get('latest_close'))
print('Created at:', data.get('created_at'))

preds = data.get('predictions', [])
print(f'\nPredictions ({len(preds)} total):')
for i, p in enumerate(preds[:5]):
    print(f'{i+1}. {p["article_title"][:50]}...')
    print(f'   Predicted: ₹{p["predicted_price"]:.2f}')
    print(f'   Change: {p["predicted_change_percent"]:.2f}%')
    # Calculate expected price
    latest_close = data.get('latest_close', 0)
    expected = latest_close * (1 + p["predicted_change_percent"] / 100) if latest_close else 0
    print(f'   Expected: ₹{latest_close:.2f} × (1 + {p["predicted_change_percent"]:.2f}%) = ₹{expected:.2f}')
    print(f'   Match: {abs(p["predicted_price"] - expected) < 1}')
    print()
