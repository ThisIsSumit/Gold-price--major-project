import requests

data = requests.get('http://127.0.0.1:8001/api/v1/snapshot/live?force_retrain=true', timeout=120).json()
latest = data['market_features']['latest_close']
summary = data['summary']

print(f'Latest close: ₹{latest:.2f}')
print(f'Summary predicted price: ₹{summary["predicted_price"]:.2f}')
print(f'Summary predicted change: {summary["predicted_change_pct"]:.4f}%')

articles = data.get('articles', [])
print(f'Articles: {len(articles)}')
print()

for idx, article in enumerate(articles[:3], 1):
    price = article['Predicted Price']
    change = article['Predicted Change %']
    expected = latest * (1 + change / 100)
    print(f'{idx}. {article["title"][:60]}')
    print(f'   Predicted Price: ₹{price:.2f}')
    print(f'   Predicted Change: {change:.4f}%')
    print(f'   Formula Check: ₹{latest:.2f} × (1 + {change:.4f}%) = ₹{expected:.2f}')
    print(f'   Match: {abs(price - expected) < 1}')
    print()
