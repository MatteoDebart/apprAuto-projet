import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from format_data import CATEGORICAL_COL, NUMERICAL_COL
from sklearn.linear_model import SGDRegressor

def plot_distribution(column):

    plt.figure(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(column):
        sns.histplot(column, kde=True, bins=10)
    else:
        sns.countplot(x=column)

    plt.ylabel('Frequency')
    plt.show()


def get_numerical_features(Db):
    weld_columns = [col for col in Db.columns if col.startswith("Type of weld_")]
    numerical_features = list(set(Db.columns) - set(weld_columns) -
                              set(CATEGORICAL_COL) - set(["output"]))
    return numerical_features

def get_categirical_features(Db):
    numerical_features = list(set(Db.columns) -
                              set(NUMERICAL_COL) - set(["output"]))
    return numerical_features