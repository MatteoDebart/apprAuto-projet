import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from format_data import CATEGORICAL_COL, NUMERICAL_COL
import joblib


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


def get_numerical_features(Db: pd.DataFrame):
    weld_columns = [
        col for col in Db.columns if col.startswith("Type of weld_")]
    numerical_features = list(set(Db.columns) - set(weld_columns) -
                              set(CATEGORICAL_COL) - set(["output"]))
    return numerical_features


def get_categorical_features(Db: pd.DataFrame):
    categorical_features = list(set(Db.columns) -
                                set(NUMERICAL_COL) - set(["output"]))
    return categorical_features


def get_weld(Db: pd.DataFrame):
    # Identify columns that start with "Type of weld_"
    weld_columns = [
        col for col in Db.columns if col.startswith("Type of weld_")]

    def extract_weld_type(row):
        # Iterate through the weld columns to find the type with value 1
        for weld in weld_columns:
            if row[weld] == 1:
                return weld.replace("Type of weld_", "")  # Remove the prefix
        return 'Other'  # Return 'Other' if no type is found

    # Apply the extract_weld_type function to each row in the DataFrame
    return Db.apply(extract_weld_type, axis=1)


def split_target_from_dataset(Db, target='output'):  # Remove target col
    y = Db[target]
    X = Db.drop(columns=[target])
    return X, y


def save_model(model, model_name):
    joblib_file = f"models/{model_name}.pkl"
    joblib.dump(model, joblib_file)
    print(f"Model saved as: {joblib_file}")
