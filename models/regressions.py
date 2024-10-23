from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
import itertools

import sys, os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))

from format_data import create_dataframe
from preprocess import preprocess_supervised, OutputColumn
from evaluation import cross_validation

def split_target_from_dataset(Db, target='output'):  # Remove target col
    y = Db[target]
    X = Db.drop(columns=[target])
    return X, y

def best_parameters_tuple(D):
    for key in D:
        if isinstance(D[key],dict):
            D[key] = best_parameters_tuple(D[key])
        else:
            D[key] = [D[key]]
        
    best_parameter = min(D, key= lambda x:abs(D[x][-1]))
    return [best_parameter, *D[best_parameter]]

def evaluate_regression(pipeline:Pipeline, param_grid:dict, X_train, X_test, y_train, y_test, Db_keys):

    '''
    This function finds the best feature subset to train the pipeline, plots the "score VS nb_features" curve and the "Predicted vs Actual" scatter plot and prints the 
    cross-validation evaluation of the found best model.

    Inputs:
        - pipeline:         the pipeline to study
        - param_grid:       the hyperparameters dictionary to test in the Grid Search training
        - X_train:          the training data
        - X_test:           the testing data
        - y_train:          the testing labels
        - y_test:           the testing labels
        - Db_keys:          the list of features names
    
    Outputs:
        None
    '''

    ################ Find best features subset #################

    print("Performing Evaluation with Cross-Validation")
    best_parameters_ids = [0]
    features_list = []
    best_scoring_list = []

    while len(best_parameters_ids) < X_train.shape[1]:          #At each step, we will allow the subset to delete 1 element and add the best pair (allowing to add the deleted element if it is relevant)
        scoring_dict = {}

        for k in best_parameters_ids:
            scoring_dict[k] = {}
            temp_best_parameters_ids = list(set(best_parameters_ids) - set([k]))
            temp_exploration = list(set(range(X_train.shape[1])) - set(temp_best_parameters_ids))

            for i,j in list(itertools.combinations(temp_exploration, 2)):
                grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                X_train_temp = np.take(X_train, temp_best_parameters_ids + [i,j], 1)
                grid_search.fit(X_train_temp, y_train)
                scoring_dict[k].update({i: {j : grid_search.best_score_}})

        [k,i,j,score] = best_parameters_tuple(scoring_dict)     #Find best choice of deletion/add

        best_parameters_ids = list(set(best_parameters_ids) - set([k])) +[i,j]
        features_list.append(best_parameters_ids)
        best_scoring_list.append(score)
        print(len(best_parameters_ids), '/', X_train.shape[1])

    best_scoring_list = np.abs(np.array(best_scoring_list))

    best_score_id = 0
    best_score = best_scoring_list[0]

    for id in range(1, len(best_scoring_list)):

        if best_scoring_list[id] <= 0.95*best_score:            #Choose the most simple model to have significantly better performances than the previous one.
            best_score_id = id
            best_score = best_scoring_list[id]

    optimal_features = features_list[best_score_id]

    plt.plot(best_scoring_list)
    plt.xlabel("nb_features")
    plt.ylabel("Validation MSE")
    plt.show()

    print("Best features: ", [Db_keys[x] for x in optimal_features])
    
    ################ Evaluating model on the best subset #################

    X_train_temp = np.take(X_train, optimal_features, 1)
    X_test_temp = np.take(X_test, optimal_features, 1)
    grid_search = GridSearchCV(pipeline, {}, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    grid_search.fit(X_train_temp, y_train)                      #Train the model on the best subset

    test_score = grid_search.score(X_test_temp, y_test)

    y_pred = grid_search.predict(X_test_temp)

    ############# Plot the predicted values against the actual values #############
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predicted vs Actual')                                           # Scatter plot of predicted vs actual
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')   # Line for perfect predictions
    
    # Set equal scaling
    plt.xlim(min(y_test), max(y_test))
    plt.ylim(min(y_test), max(y_test))
    plt.gca().set_aspect('equal', adjustable='box')             # Set equal aspect ratio

    plt.title('Predicted vs Actual values of y_test')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.legend()
    plt.grid()
    plt.show()

    print(cross_validation(pipeline, np.array(X_train_temp), np.array(y_train)))

def evaluate_regression_no_plots(pipeline, param_grid, X_train, y_train):

    '''
    This function finds the best feature subset to train the pipeline and returns its mean square error

    Inputs:
        - pipeline:         the pipeline to study
        - param_grid:       the hyperparameters dictionary to test in the Grid Search training
        - X_train:          the training data
        - y_train:          the testing labels
    
    Outputs:
        - Best_score:       the cross-validation score of the model trained on the best features subset.
    '''
    
    best_parameters_ids = [0]
    best_scoring_list = []
    features_list = []

    while len(best_parameters_ids) < X_train.shape[1]:          #At each step, we will allow the subset to delete 1 element and add the best pair (allowing to add the deleted element if it is relevant)
        scoring_dict = {}
        for k in best_parameters_ids:
            scoring_dict[k] = {}
            temp_best_parameters_ids = list(set(best_parameters_ids) - set([k]))
            temp_exploration = list(set(range(X_train.shape[1])) - set(temp_best_parameters_ids))
            for i,j in list(itertools.combinations(temp_exploration, 2)):
                grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
                X_train_temp = np.take(X_train, temp_best_parameters_ids + [i,j], 1)
                grid_search.fit(X_train_temp, y_train)
                scoring_dict[k].update({i: {j : grid_search.best_score_}})

        [k,i,j,score] = best_parameters_tuple(scoring_dict)
        best_parameters_ids = list(set(best_parameters_ids) - set([k])) +[i,j]
        features_list.append(best_parameters_ids)
        best_scoring_list.append(score)
    
    best_scoring_list = np.abs(np.array(best_scoring_list))

    best_score_id = 0
    best_score = best_scoring_list[0]
    for id in range(1, len(best_scoring_list)):
        if best_scoring_list[id] <= 0.95*best_score:            #Choose the most simple model to have significantly better performances than the previous one.
            best_score_id = id
            best_score = best_scoring_list[id]
            
    optimal_features = features_list[best_score_id]

    print("Best features: ", optimal_features)

    return best_scoring_list[best_score_id]

def seek_best_degree(n, X_train, y_train):

    '''
    This function finds the best degree for the polynomial regression based on the best score obtained using the evaluate_regression_no_plots function.
    It also plots the "MSE VS degree" curve.

    Inputs:
        - n:            The highest polynomial degree to test
        - X_train:      The training data
        - y_train:      The training labels
    
    Outputs:
        - best_degree:  The best degree found with this method
    
    '''

    list_scores = []
    min_score = np.inf

    for i in range(1,n+1):
        pipeline_linreg = Pipeline([('preprocesser', PolynomialFeatures(i)),('model', SGDRegressor())])
        score = evaluate_regression_no_plots(pipeline_linreg, {}, X_train, y_train)
        list_scores.append(score)
        if score <= min_score:
            min_score = score
            best_degree = i
    
    plt.plot(list(range(1,n+1)),list_scores)
    plt.title("MSE VS degree of the polynomial regression")
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.show()

    return best_degree

if __name__ == "__main__":
    data_file_path="welddb/welddb.data"
    target = OutputColumn.yield_strength

    print("Loading the dataset")
    Db = create_dataframe(data_file_path)
    print("Preprocessing")
    Db = preprocess_supervised(Db, target)

    print("Split target from the dataset")
    X, y = split_target_from_dataset(Db)

    print("Train/Test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    
    degree = seek_best_degree(4, X_train, y_train)
    
    print("Creating Pipelines")
    pipeline_linreg = Pipeline([('preprocesser', PolynomialFeatures(degree)),('model', SGDRegressor())])

    evaluate_regression(pipeline_linreg, {}, X_train, X_test, y_train, y_test, Db.keys())