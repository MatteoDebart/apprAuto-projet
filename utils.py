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


def get_corr(table: pd.DataFrame):
    # Initialize the result dictionary
    col_info = {
    }

    for col in table.columns:
        column = table[col]
        missing_ratio = column.isnull().mean()

        col_info[col] = {'missing_ratio': missing_ratio}

        valid_rows = table[[col, 'output']].dropna()

        if not valid_rows.empty:
            correlation = valid_rows.corr()
            col_info[col]['correlation_with_output'] = correlation.values[0, 1]
        else:
            # No valid data to correlate
            col_info[col]['correlation_with_output'] = np.nan

    return col_info

def get_numerical_features(Db:pd.DataFrame):
    weld_columns = [col for col in Db.columns if col.startswith("Type of weld_")]
    numerical_features = list(set(Db.columns) - set(weld_columns) -
                              set(CATEGORICAL_COL) - set(["output"]))
    return numerical_features

def get_categorical_features(Db:pd.DataFrame):
    categorical_features = list(set(Db.columns) -
                              set(NUMERICAL_COL) - set(["output"]))
    return categorical_features

