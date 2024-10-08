# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

from format_data import create_dataframe
from preprocess import preprocess_supervised, OutputColumn


def split_target_from_dataset(Db, target='output'):  # Remove target col
    y = Db[target]
    X = Db.drop(columns=[target])
    return X, y


def save_model(pipeline, model_filepath):
    joblib.dump(pipeline, model_filepath)


if __name__ == "__main__":
    model_file_path = 'models/random_forest_model.pkl'
    target = OutputColumn.yield_strength

    print("Loading the dataset")
    Db = create_dataframe()
    print("Preprocessing")
    Db = preprocess_supervised(Db, target)

    print("Split target from the dataset")
    X, y = split_target_from_dataset(Db)

    print("Train/Test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21)

    print("Creating Pipeline")
    pipeline = Pipeline([
        ('model', RandomForestRegressor(n_estimators=100, random_state=17))
    ])

    print("Training")
    pipeline.fit(X_train, y_train)

    print("Evaluation on test set")
    # y_pred = pipeline.predict(X_test)
    print("Score: ", pipeline.score(X_test, y_test))

    print("Saving the model")
    save_model(pipeline, model_file_path)
