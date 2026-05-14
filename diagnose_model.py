import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

from gold_app.model import load_model
from gold_app.config import MODEL_PATH
import joblib

print('=' * 60)
print('MODEL ARTIFACT DIAGNOSTIC')
print('=' * 60)

print(f'\nArtifact path: {MODEL_PATH}')
print(f'Exists: {MODEL_PATH.exists()}')

if MODEL_PATH.exists():
    import os
    import time
    mtime = os.path.getmtime(MODEL_PATH)
    size = os.path.getsize(MODEL_PATH)
    print(f'Size: {size} bytes')
    print(f'Modified: {time.ctime(mtime)}')
    
    # Load raw joblib
    print('\nLoading artifact...')
    raw_bundle = joblib.load(MODEL_PATH)
    print(f'Bundle type: {type(raw_bundle).__name__}')
    
    if hasattr(raw_bundle, 'metrics'):
        print(f'Metrics: {raw_bundle.metrics}')
    
    # Check regressor coefficients scale
    if hasattr(raw_bundle, 'regressor'):
        reg = raw_bundle.regressor
        print(f'\nRegressor type: {type(reg).__name__}')
        
        if hasattr(reg, 'steps'):
            for step_name, step_obj in reg.steps:
                print(f'  Step: {step_name} ({type(step_obj).__name__})')
                if hasattr(step_obj, 'coef_'):
                    coef = step_obj.coef_
                    print(f'    Coefficients shape: {coef.shape}')
                    print(f'    Coef range: {coef.min():.6f} to {coef.max():.6f}')
    
    print('\nTraining data config:')
    from gold_app.config import TRAINING_MARKET_PATH, TRAINING_NEWS_PATH
    print(f'  Market: {TRAINING_MARKET_PATH.name}')
    print(f'  News: {TRAINING_NEWS_PATH.name}')
else:
    print('Artifact not found!')
