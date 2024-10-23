from sklearn.ensemble import RandomForestRegressor

import numpy as np
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))

from format_data import create_dataframe
from preprocess import preprocess_supervised, OutputColumn
from plots import plot_y_pred
from models.evaluation import evaluation
from train_model import split_target_from_dataset, save_model


def select_model(X, y):
    """
    Selects the best model hyperparameters based on the provided data.
    """

    param_grid = {
        'n_estimators': [100, 125, 150, 175, 200, 225, 250, 275, 300],
        'max_depth': [5, 10, 15, 20, 25, 30],
        'max_features': [None, 'sqrt', 'log2', 10, 7],
        # 10 = n_features/2 ; 7 = n_features/3
    }

    best_params = None
    best_oob_score = -1
    n_iter_search = 1
    print(f"Trying {n_iter_search} combinations of parameters")

    for i in range(n_iter_search):

        params = {key: np.random.choice(values)
                  for key, values in param_grid.items()}
        model = RandomForestRegressor(
            oob_score=True, random_state=42, **params)
        model.fit(X, y)

        oob_score = model.oob_score_

        if oob_score > best_oob_score:
            best_oob_score = oob_score
            best_params = params

    return best_params


def complete_pipeline(X, y):
    """
    Completes the entire model pipeline, including model selection, prediction, evaluation, 
    and saving the best model.
    """

    print(f"Selecting best Random Forest hyper-parameters")
    best_params = select_model(X, y)
    print(f"Choose parameters: {best_params}")

    best_model = RandomForestRegressor(
        oob_score=True,  # Enable OOB score
        random_state=17,
        bootstrap=True,  # Bootstrap sampling required for OOB
        n_estimators=best_params['n_estimators'],
        max_features=best_params['max_features'],
        max_depth=best_params['max_depth'],
    )

    best_model.fit(X, y)

    # Plotting using the Out of the Bag predictions
    plot_y_pred(y, best_model.oob_prediction_)

    print("Evaluation results:")
    print(evaluation(best_model, X, y, use_oob=True))

    save_model(best_model, 'random_forest')


if __name__ == "__main__":

    data_file_path = "welddb/welddb.data"
    target = OutputColumn.yield_strength

    print("[STEP 1] Loading the dataset..")
    Db = create_dataframe(data_file_path)
    print("[STEP 2] Preprocessing..")
    Db = preprocess_supervised(Db, target)
    X, y = split_target_from_dataset(Db)

    print("[STEP 3] Parameters selection..")
    complete_pipeline(X, y)
