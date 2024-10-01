import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def convert_less_than(value):
    """Converts less-than instances with real numbers respecting the condition."""
    
    if isinstance(value, str) and value.startswith('<'):
        try:
            number = float(value[1:])
            return np.random.uniform(0, number)
        except ValueError:
            return value
    return value


def plot_distribution(column):
    """Plots the distribution of a feature."""

    plt.figure(figsize=(8, 4))
    
    if pd.api.types.is_numeric_dtype(column):
        sns.histplot(column.dropna(), kde=False)
    else: 
        sns.countplot(y=column)
    
    plt.show()