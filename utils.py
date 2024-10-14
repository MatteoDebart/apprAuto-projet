import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from format_data import CATEGORICAL_COL, NUMERICAL_COL

def plot_distribution(column):

    plt.figure(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(column):
        sns.histplot(column, kde=True, bins=10)
    else:
        sns.countplot(x=column)

    plt.ylabel('Frequency')
    plt.show()


def get_numerical_features(Db:pd.DataFrame):
    weld_columns = [col for col in Db.columns if col.startswith("Type of weld_")]
    numerical_features = list(set(Db.columns) - set(weld_columns) -
                              set(CATEGORICAL_COL) - set(["output"]))
    return numerical_features

def get_categorical_features(Db:pd.DataFrame):
    numerical_features = list(set(Db.columns) -
                              set(NUMERICAL_COL) - set(["output"]))
    return numerical_features

def weld_type(Db:pd.DataFrame):
    weld_columns = [col for col in Db.columns if col.startswith("Type of weld_")]
    def weld(row):
        for weld, value in row.items():
            if value:
                return weld.replace("Type of weld_", "")  # Remove the prefix
        return 'Other'
    
    return Db[weld_columns].apply(weld, axis=1)

