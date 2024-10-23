from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from models.boosting_techniques import complete_pipeline
from format_data import create_dataframe
from preprocess import OutputColumn
from preprocess_semi import preprocess_semisupervised
from utils import split_target_from_dataset


class XGBoostConfigSemiSupervised:
    """XGBoost model configuration."""

    def __init__(self):
        self.model_name = 'xgboost_semisupervised'
        self.model = XGBRegressor()
        self.param_grid = {
            'model__n_estimators': [100, 300, 500, 700, 900],
            'model__max_depth': [3, 6, 9, 11, 13],
            'model__eta': [0.01, 0.1, 0.2],
            'model__alpha': [0, 0.1, 0.2],
        }


if __name__ == "__main__":

    data_file_path = "welddb/welddb.data"
    target = OutputColumn.yield_strength

    print("[STEP 1] Loading the dataset..")
    Db_complete = create_dataframe(data_file_path)

    print("[STEP 2] Preprocessing and Pseudo-labeling..")
    Db_complete = preprocess_semisupervised(Db_complete, target, model=Models.xgboost)
    X, y = split_target_from_dataset(Db_complete)


    print("[STEP 3] Train/Test split..")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21)

    # XGBOOST
    print("[STEP 4] Model selection..")
    xgboost_config = XGBoostConfigSemiSupervised()
    complete_pipeline(X_train, X_test, y_train, y_test, xgboost_config)

