import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from preprocess import feature_decision
from sklearn.cross_decomposition import PLSRegression

def column_info(col_title, table):
    col = table[col_title]
    missing_ratio = col.isnull().mean()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(col):
        sns.histplot(col.dropna(), kde=True)
    else: # the variable is categorical
        sns.countplot(y=col, order=col.value_counts().index)
    
    plt.title(f"{col_title}: {missing_ratio:.2f}% missing")
    plt.xlabel(col_title)
    
    plt.show()


def plot_completeness_vs_corr(col_info, feature_decision=feature_decision):
    # Initialize lists to store x (completeness) and y (correlation) values
    x_completeness = []
    y_correlation = []
    column_names = []


    # Loop through the columns in col_info
    for col, info in col_info.items():
        if 'correlation_with_output' in info:  # Only consider columns with calculated correlation
            if col=='output':
                continue

            completeness = 1 - info['missing_ratio'] 
            correlation = abs(info['correlation_with_output'])
            
            x_completeness.append(completeness)
            y_correlation.append(correlation)
            column_names.append(col)

        
    # Plotting
    plt.figure(figsize=(20, 20))
    plt.scatter(x_completeness, y_correlation, color='b', marker='o')
    plt.ylim(0, 0.5)
    
    # Label each point with the column name
    for i, txt in enumerate(column_names):
        plt.annotate(txt, (x_completeness[i], y_correlation[i]), fontsize=12, ha='right')

    if feature_decision is not None:
        comp = np.linspace(min(x_completeness), max(x_completeness), 100)
        corr = feature_decision(comp)
        plt.plot(comp, corr, '-r')

    # Labels and title (in French)
    plt.xlabel("Complétude de la colonne", fontsize=12)
    plt.ylabel("Corrélation avec l'output en valeur absolue", fontsize=12)
    plt.title("Complétude vs Corrélation", fontsize=14)

    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

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