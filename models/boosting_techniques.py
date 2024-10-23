from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from format_data import create_dataframe
from preprocess import preprocess_supervised, OutputColumn
from plots import plot_y_pred
from models.evaluation import evaluation
from utils import save_model, split_target_from_dataset


class XGBoostConfigSupervised:
    """XGBoost model configuration."""

    def __init__(self):
        self.model_name = 'xgboost_supervised'
        self.model = XGBRegressor()
        self.param_grid = {
            'model__n_estimators': [600, 650, 700, 750, 800],
            'model__max_depth': [3, 4, 5, 6],
            'model__eta': [0.06, 0.08, 0.1, 0.12, 0.14],
            'model__alpha': [0, 0.1, 0.2]
        }


class LGBMConfigSupervised:
    """LightGBM model configuration."""

    def __init__(self):
        self.model_name = 'lgbm_supervised'
        self.model = LGBMRegressor(silent=True)
        self.param_grid = {
            'model__n_estimators': [100, 300, 500, 600],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__num_leaves': [15, 30, 50, 100]
        }


def select_model(X_train, y_train, model_config):
    """
    Selects the best model based on the provided training data and model configuration.
    """

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_config.model)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_config.param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters for {model_config.model_name}: {best_params}")
    best_model = grid_search.best_estimator_

    return best_model


def complete_pipeline(X_train, X_test, y_train, y_test, model_config):
    """
    Completes the entire model pipeline, including model selection, prediction, evaluation, 
    and saving the best model.
    """

    print(f"Selecting best {model_config.model_name} model")
    best_model = select_model(X_train, y_train, model_config)

    y_pred = best_model.predict(X_test)
    plot_y_pred(y_test, y_pred)

    print("Evaluation results:")
    print(evaluation(best_model, X_test, y_test))

    save_model(best_model, model_config.model_name)


if __name__ == "__main__":

    data_file_path = "welddb/welddb.data"
    target = OutputColumn.yield_strength

    print("[STEP 1] Loading the dataset..")
    Db = create_dataframe(data_file_path)
    print("[STEP 2] Preprocessing..")
    Db = preprocess_supervised(Db, target)
    X, y = split_target_from_dataset(Db)

    print("[STEP 3] Train/Test split..")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21)

    # XGBOOST
    print("[STEP 4] Model selection..")
    xgboost_config = XGBoostConfigSupervised()
    complete_pipeline(X_train, X_test, y_train, y_test, xgboost_config)

    # # LGBM
    # lgbm_config = LGBMConfig()
    # complete_pipeline(X_train, X_test, y_train, y_test, lgbm_config)
