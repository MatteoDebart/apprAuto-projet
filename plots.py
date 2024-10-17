import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from preprocess import feature_decision

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


def plot_y_pred(y, y_pred, weld_types=None):
    plt.figure(figsize=(8, 6))
    
    if weld_types is not None:
        # If weld is provided, create a scatter plot with different colors for each weld type
        welds = set(weld_types)
        for w in welds:
            mask = (weld_types == w)
            plt.scatter(y[mask], y_pred[mask], label=f"Weld: {w}", alpha=0.7)
    else:
        # If weld is not provided, plot a regular scatter plot
        plt.scatter(y, y_pred, alpha=0.7, label='Predictions')

    # Plot the perfect y = y_pred line
    min_val = min(np.min(y), np.min(y_pred))
    max_val = max(np.max(y), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y = y_pred)')
    
    # Add labels and legend
    plt.xlabel('True Values (y)')
    plt.ylabel('Predicted Values (y_pred)')
    plt.title('True vs Predicted Values')
    
    if weld_types is not None:
        plt.legend()
    
    plt.grid(True)
    plt.show()