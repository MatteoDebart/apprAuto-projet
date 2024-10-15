from typing import Optional
import pandas as pd
import numpy as np
from enum import Enum
from format_data import MECHANICAL_PROPERTIES, CATEGORICAL_COL, NUMERICAL_COL
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils import get_numerical_features, impute_categorical
from format_data import create_dataframe


class OutputColumn(Enum):
    elongation = "Elongation / %"
    yield_strength = "Yield strength / MPa"
    ch_imp_toughness = "Charpy impact toughness / J"


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


def feature_decision(completeness):
    points = np.array([(0, 0.45), (0.4, 0.22778), (0.6, 0.05)])
    x_points = points[:, 0]
    y_points = points[:, 1]
    coefficients = np.polyfit(x_points, y_points, 2)
    f = np.vectorize(lambda c: max(0.05, np.polyval(coefficients, c)))
    return f(completeness)


def feature_selection(col_info, feature_decision=feature_decision):
    features = ['output']
    # Loop through the columns in col_info
    for col, info in col_info.items():
        if 'correlation_with_output' in info:  # Only consider columns with calculated correlation
            if col == 'output':
                continue

            completeness = 1 - info['missing_ratio']
            correlation = abs(info['correlation_with_output'])

            if feature_decision is not None and feature_decision(completeness) <= correlation:
                features.append(col)

    return features


def handle_outliers(Db: pd.DataFrame, iqr_features: list[str] = ["Manganese concentration / (weight%)", "Carbon concentration / (weight%)", "Niobium concentration / parts per million by weight", "Molybdenum concentration / (weight%)"]):
    '''Implement IQR on the selected featurs to remove outliers
    Db: the dataset
    iqr_features: the list of features on which to apply IQR 
    '''
    num_features = get_numerical_features(Db)
    for col in num_features:
        if col in iqr_features:
            non_nan_data = Db[col].dropna()

            # Calculate Q1, Q3 and IQR based on non-NaN values
            Q1 = non_nan_data.quantile(0.25)
            Q3 = non_nan_data.quantile(0.75)
            IQR = Q3 - Q1

            # Define lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            for _, row in Db.iterrows():
                if not np.isnan(row[col]) and (lower_bound > row[col] or row[col] > upper_bound):
                    row[col] = np.nan
    return Db


def imputation(Db: pd.DataFrame):
    # numerical columns
    num_columns = list(set(Db.columns) -
                       set(CATEGORICAL_COL) - set(["output"]))
    iterative_imputer = IterativeImputer(
        random_state=42, sample_posterior=True)
    Db[num_columns] = iterative_imputer.fit_transform(Db[num_columns])
    # categorical columns
    cat_columns = list(set(Db.columns) - set(NUMERICAL_COL) - set(["output"]))
    Db[cat_columns] = impute_categorical(Db, cat_columns)
    Db[cat_columns] = Db[cat_columns].apply(lambda x: x >= 0.5)
    return Db


def preprocess_supervised(Db: pd.DataFrame, output_col: OutputColumn, PLS=False, n_components=5):
    Db = Db.rename(columns={output_col.value: 'output'})
    features = list(set(Db.columns)-set(MECHANICAL_PROPERTIES))

    # For the supervised approach we only keep the rows with an output
    reduced_Db = Db.dropna(subset=['output'])
    print(
        f"We retain only the rows with output values {output_col.value}, that is {100*len(reduced_Db)/len(Db):2f}% of the dataset")

    # We keep the rows this high correlation and completeness
    col_info = get_corr(reduced_Db[features])
    features = feature_selection(col_info)

    Db = handle_outliers(Db[features])
    # We do the imputation with as many rows as possible
    imputed_Db = imputation(Db)
    Db = imputed_Db.dropna(subset=['output'])

    # Must be scaled before PCA
    # Will ask the lecturer tomorrow about the PLS criteria r² or adjusted

    return Db


if __name__ == "__main__":
    db = create_dataframe()
    processed_db = preprocess_supervised(db, OutputColumn.yield_strength)
