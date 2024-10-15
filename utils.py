import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from format_data import CATEGORICAL_COL
import miceforest as mf


def convert_less_than(value):
    if isinstance(value, str) and value.startswith('<'):
        try:
            number = float(value[1:])
            return np.random.uniform(0, number)
        except ValueError:
            return value
    return value


def plot_distribution(column):

    plt.figure(figsize=(6, 4))

    if pd.api.types.is_numeric_dtype(column):
        sns.histplot(column, kde=True, bins=10)
    else:
        sns.countplot(x=column)

    plt.ylabel('Frequency')
    plt.show()


def get_numerical_features(Db):
    numerical_features = list(set(Db.columns) -
                              set(CATEGORICAL_COL) - set(["output"]))
    return numerical_features


def impute_categorical(Db, categorical_columns_to_impute):
    
    kds = mf.ImputationKernel(
                        Db,
                        variable_schema=categorical_columns_to_impute,
                        random_state=1991
                        )
    kds.mice(2)
    imputed_Db = kds.complete_data()

    return imputed_Db[categorical_columns_to_impute]

