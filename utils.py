import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(column):
    """Plots the distribution of a feature."""

    plt.figure(figsize=(8, 4))
    
    if pd.api.types.is_numeric_dtype(column):
        sns.histplot(column.dropna(), kde=False)
    else: 
        sns.countplot(y=column)
    
    plt.show()