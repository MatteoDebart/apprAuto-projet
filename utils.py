import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< HEAD
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

=======
from format_data import CATEGORICAL_COL, NUMERICAL_COL
>>>>>>> bc01f2df7b4655598facbdd648532d8e978b9b9c

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

<<<<<<< HEAD

def impute_categorical(Db, categorical_columns_to_impute):
    
    kds = mf.ImputationKernel(
                        Db,
                        variable_schema=categorical_columns_to_impute,
                        random_state=1991
                        )
    kds.mice(2)
    imputed_Db = kds.complete_data()

    return imputed_Db[categorical_columns_to_impute]

=======
def get_categirical_features(Db):
    numerical_features = list(set(Db.columns) -
                              set(NUMERICAL_COL) - set(["output"]))
    return numerical_features
>>>>>>> bc01f2df7b4655598facbdd648532d8e978b9b9c
