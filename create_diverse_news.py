import pandas as pd
import random

# Load GOLDBEES dates
goldbees = pd.read_csv('GOLDBEES_NS_training_data.csv')
goldbees['Dates'] = pd.to_datetime(goldbees['Date'], format='%m/%d/%Y')

# News headlines pool for variety
headlines = [
    'Gold prices surge as market volatility increases',
    'RBI policy announcement impacts gold trading',
    'Indian rupee movement affects gold imports',
    'Global gold demand remains strong despite headwinds',
    'MCX gold futures rally on investment demand',
    'Gold ETF flows increase as safe haven demand grows',
    'Inflation concerns support precious metal prices',
    'Dollar weakness boosts gold attractiveness globally',
    'Central bank purchases push gold prices higher',
    'Economic uncertainty drives gold buying interest',
    'Geopolitical tensions elevate gold risk premium',
    'Interest rate expectations impact gold demand',
    'Jewelry demand in India remains steady',
    'Industrial gold consumption shows mixed trends',
    'Technical analysis suggests continued gold strength',
    'Gold mining stocks outperform sector averages',
    'Portfolio diversification demands push bullion sales',
    'Asian gold markets see strong local demand',
    'Safe haven flows into precious metals accelerate',
    'Market forecasters raise gold price targets'
]

news_data = []
for idx, row in goldbees.iterrows():
    sentiment_choice = random.choice(['positive', 'neutral', 'negative'])
    direction_choice = random.choice([0, 1, 2])  # 0=up, 1=constant, 2=down
    
    news_data.append({
        'Dates': row['Dates'].strftime('%d-%m-%Y'),
        'URL': f'https://gold.example.com/news/{idx}',
        'News': headlines[idx % len(headlines)],
        'Price Direction Up': 1 if direction_choice == 0 else 0,
        'Price Direction Constant': 1 if direction_choice == 1 else 0,
        'Price Direction Down': 1 if direction_choice == 2 else 0,
        'Asset Comparision': 0,
        'Past Information': 1,
        'Future Information': 0,
        'Price Sentiment': sentiment_choice
    })

news_df = pd.DataFrame(news_data)
news_df.to_csv('GOLDBEES_training_news.csv', index=False)
print('Created diverse news with', len(news_df), 'rows')
print('Unique headlines:', len(set([n['News'] for n in news_data])))
print('Sample sentiments:', news_df['Price Sentiment'].value_counts().to_dict())
