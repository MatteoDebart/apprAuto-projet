import pandas as pd

import numpy as np
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import KFold

def cross_validation(model, X, y, k=5):
    # Initialize lists to store the results
    mse_scores = []
    r2_scores = []
    bias_scores = []
    
    # KFold splits
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    predictions_all_folds = []  # To store all predictions for variance calculation
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate MSE and R² for the current fold
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate bias
        bias = np.mean(y_pred - y_test)
        
        # Store all predictions for variance calculation
        predictions_all_folds.append(y_pred)
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        bias_scores.append(bias)
    
    # Variance of predictions across all folds
    predictions_all_folds = np.concatenate(predictions_all_folds)
    variance = np.var(predictions_all_folds)

    results = {
        'MSE': np.mean(mse_scores),
        'R²': np.mean(r2_scores),
        'Bias': np.mean(bias_scores),
        'Variance': variance
    }
    
    return results


