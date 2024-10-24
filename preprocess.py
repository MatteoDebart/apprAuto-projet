from typing import Optional
import pandas as pd
import numpy as np
from enum import Enum
from format_data import MECHANICAL_PROPERTIES, CATEGORICAL_COL, NUMERICAL_COL
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import miceforest as mf

from utils import get_numerical_features, get_categorical_features, get_corr
from format_data import create_dataframe


class OutputColumn(Enum):
    elongation = "Elongation / %"
    yield_strength = "Yield strength / MPa"
    ch_imp_toughness = "Charpy impact toughness / J"


def feature_decision(completeness):
    '''
    Creates a function of the completeness. Features with features(completeness)>correlation won't be selected
    '''
    points = np.array([(0, 0.45), (0.4, 0.22778), (0.6, 0.05)])
    x_points = points[:, 0]
    y_points = points[:, 1]
    coefficients = np.polyfit(x_points, y_points, 2)
    f = np.vectorize(lambda c: max(0.05, np.polyval(coefficients, c)))
    return f(completeness)


def feature_selection(col_info, feature_decision=feature_decision):
    '''
    Select the features with correlation and completeness high enough
    '''
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


def imputation(Db: pd.DataFrame, numerical=True, categorical=True):
    '''
    Imputes the missing values in the dataset. The process is different for numerical and categorical columns.
    '''
    # numerical columns
    if numerical:
        num_columns = get_numerical_features(Db)
        iterative_imputer = IterativeImputer(
            random_state=42, sample_posterior=True)
        Db[num_columns] = iterative_imputer.fit_transform(Db[num_columns])
    # categorical columns
    if categorical:
        cat_columns = get_categorical_features(Db)
        kds = mf.ImputationKernel(Db,
                                  variable_schema=cat_columns,
                                  random_state=1991)
        kds.mice(2)
        imputed_Db = kds.complete_data()
        Db[cat_columns] = imputed_Db[cat_columns].apply(lambda x: x >= 0.5)
    return Db


def preprocess_supervised(Db: pd.DataFrame, output_col: OutputColumn, all_welds=False):
    '''
    Preprocess the dataset: Handle outliers, impute the features, and select those which will be in the final dataset
    '''
    Db_copy = Db.copy()
    Db_copy = Db_copy.rename(columns={output_col.value: 'output'})

    # Outliers and scaling
    Db_copy = handle_outliers(Db_copy)
    scaler = StandardScaler()
    scaled_feature = get_numerical_features(Db_copy)+['output']
    Db_copy[scaled_feature] = scaler.fit_transform(Db_copy[scaled_feature])

    # we look at the correlation with the output and the columns with the least NaN values where the output is present
    reduced_Db = Db_copy.dropna(subset=['output'])
    print(
        f"We retain only the rows with output values {output_col.value}, that is {100*len(reduced_Db)/len(Db_copy):2f}% of the dataset")

    # We keep the rows this high correlation and completeness
    features = list(set(Db_copy.columns)-set(MECHANICAL_PROPERTIES))
    col_info = get_corr(reduced_Db[features])
    features = feature_selection(col_info)
    if all_welds:
        weld_columns = [
            col for col in Db_copy.columns if col.startswith("Type of weld_")]
        for col in weld_columns:
            if col not in features:
                features.append(col)

    # We do the imputation with as many rows as possible
    imputed_Db = imputation(Db_copy)
    imputed_Db[scaled_feature] = scaler.inverse_transform(
        imputed_Db[scaled_feature])

    # But for the supervised approach we only keep the rows with an output
    Db_copy = imputed_Db.dropna(subset=['output'])[features]

    return Db_copy


if __name__ == "__main__":
    Db = create_dataframe()
    processed_db = preprocess_supervised(Db, OutputColumn.yield_strength)
