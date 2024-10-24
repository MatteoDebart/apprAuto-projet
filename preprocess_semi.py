from models.regressions import *
from sklearn.linear_model import SGDRegressor
from enum import Enum
from preprocess import *
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from models.evaluation import *

class Models(Enum):
    regressions = SGDRegressor()
    randomforest = RandomForestRegressor()
    xgboost = XGBRegressor(n_estimators=700, max_depth=3, learning_rate=0.14, alpha=0.2)

    
def preprocess_semisupervised(Db: pd.DataFrame, output_col: OutputColumn, model:Models, all_welds=False):
    Db = Db.rename(columns={output_col.value: 'output'})
    model=model.value

    # Outliers and scaling
    Db=handle_outliers(Db)
    scaler = StandardScaler()
    scaled_feature = get_numerical_features(Db)+['output']
    Db[scaled_feature] = scaler.fit_transform(Db[scaled_feature])

    # we look at the correlation with the output and the columns with the least NaN values where the output is present
    reduced_Db = Db.dropna(subset=['output'])

    # We keep the rows this high correlation and completeness
    features = list(set(Db.columns)-set(MECHANICAL_PROPERTIES))
    col_info = get_corr(reduced_Db[features])
    features = feature_selection(col_info)
    if all_welds:
        weld_columns = [col for col in Db.columns if col.startswith("Type of weld_")]
        for col in weld_columns:
            if col not in features:
                features.append(col)
    

    # We do the imputation with as many rows as possible 
    imputed_Db = imputation(Db)
    imputed_Db[scaled_feature] = scaler.inverse_transform(imputed_Db[scaled_feature])

    # But for the supervised approach we only keep the rows with an output
    Db_train = imputed_Db.dropna(subset=['output'])[features]
    y=Db_train['output']
    X=Db_train.drop(['output'], axis=1)
    model.fit(X,y)
    
    Db_to_predict = imputed_Db[imputed_Db['output'].isna()][features]
    X_to_predict = Db_to_predict.drop(['output'], axis=1)

    y_to_predict = model.predict(X_to_predict)
    Db_to_predict['output'] = y_to_predict
    Db_complete = pd.concat([Db_train, Db_to_predict])

    return Db_complete

# def evaluation_semi(Db, output_col:OutputColumn, model:Models):
#     Db_complete = preprocess_semisupervised(Db, output_col, model)

#     y_complete = Db_complete['output']
#     X_complete = Db_complete.drop(['output'], axis=1)

    
#     X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.2, random_state=42)
    
#     model_instance = model.value  # Extract the model from the Enum
#     model_instance.fit(X_train, y_train)

#     return evaluation(model_instance, X_test, y_test)