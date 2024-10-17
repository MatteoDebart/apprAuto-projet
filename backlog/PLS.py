import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression

def plot_PLS(pls:PLSRegression, X, y):
    Y_pred = pls.predict(X)
    X_reduced = pls.transform(X)
    r2 = pls.score(X, y)
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot the reduced dimensions on the left
    scatter1 = ax[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
    ax[0].set_xlabel('Component 1')
    ax[0].set_ylabel('Component 2')
    ax[0].set_title('PLS Reduced Dimensions of X')
    fig.colorbar(scatter1, ax=ax[0], label='Target (y)')

    # Plot actual vs predicted values on the right
    scatter2 = ax[1].scatter(y, Y_pred, color='blue', label='Predicted vs Actual')
    ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')  # Diagonal line for perfect predictions
    ax[1].set_xlabel('Actual Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].set_title(f'PLS Regression: Actual vs Predicted (R²: {r2:.2f})')
    ax[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()