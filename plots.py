import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from preprocess import feature_decision
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



def column_info(col_title, table):
    col = table[col_title]
    missing_ratio = col.isnull().mean()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(col):
        sns.histplot(col.dropna(), kde=True)
    else: # the variable is categorical
        sns.countplot(y=col, order=col.value_counts().index)
    
    plt.title(f"{col_title}: {missing_ratio:.2f}% missing")
    plt.xlabel(col_title)
    
    plt.show()


def plot_completeness_vs_corr(col_info, feature_decision=feature_decision):
    # Initialize lists to store x (completeness) and y (correlation) values
    x_completeness = []
    y_correlation = []
    column_names = []


    # Loop through the columns in col_info
    for col, info in col_info.items():
        if 'correlation_with_output' in info:  # Only consider columns with calculated correlation
            if col=='output':
                continue

            completeness = 1 - info['missing_ratio'] 
            correlation = abs(info['correlation_with_output'])
            
            x_completeness.append(completeness)
            y_correlation.append(correlation)
            column_names.append(col)

        
    # Plotting
    plt.figure(figsize=(20, 20))
    plt.scatter(x_completeness, y_correlation, color='b', marker='o')
    plt.ylim(0, 0.5)
    
    # Label each point with the column name
    for i, txt in enumerate(column_names):
        plt.annotate(txt, (x_completeness[i], y_correlation[i]), fontsize=12, ha='right')

    if feature_decision is not None:
        comp = np.linspace(min(x_completeness), max(x_completeness), 100)
        corr = feature_decision(comp)
        plt.plot(comp, corr, '-r')

    # Labels and title (in French)
    plt.xlabel("Complétude de la colonne", fontsize=12)
    plt.ylabel("Corrélation avec l'output en valeur absolue", fontsize=12)
    plt.title("Complétude vs Corrélation", fontsize=14)

    # Show grid
    plt.grid(True)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_PLS(pls:PLSRegression, X, y):
    Y_pred = pls.predict(X)
    X_reduced = pls.transform(X)
    r2 = pls.score(X, y)
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # Plot the reduced dimensions on the left
    scatter1 = ax[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k')
    ax[0].set_xlabel('Component 1')
    ax[0].set_ylabel('Component 2')
    ax[0].set_title('PLS Reduced Dimensions of X')
    fig.colorbar(scatter1, ax=ax[0], label='Target (y)')

    # Plot actual vs predicted values on the right
    scatter2 = ax[1].scatter(y, Y_pred, color='blue', label='Predicted vs Actual')
    ax[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit')  # Diagonal line for perfect predictions
    ax[1].set_xlabel('Actual Values')
    ax[1].set_ylabel('Predicted Values')
    ax[1].set_title(f'PLS Regression: Actual vs Predicted (R²: {r2:.2f})')
    ax[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_PCA(Df:pd.DataFrame, pca_features:list, category:str = None):
    '''
    This function realises the PCA Analysis, decides how many features are relevant using the 80% of cumulative inertia property and plots the 2D graph of each pair of relevant feature.


    Input:  - Df:               The Dataframe containing the Dataset with imputed missing values
            - pca_features:     The names of the columns considered as input of the PCA
            - category:         The name of the column of the categorial feture used for colors in the final plot
    
    Returns: pca:                   The instance of PCA trained on the features
             nb_relevant_features   The number of relevant PCA features
             features_importance    The list of features in decreasing order of importance in the contribution to variance
             sim_pairs              The list of pairs of features which have a similar influence on variability (cosine similarity >=0.8)
    '''

    df_pca = (Df[pca_features] - Df[pca_features].mean())/Df[pca_features].std()
    pca = PCA(n_components=len(pca_features))
    df_principal_components = pca.fit_transform(df_pca)

    nb_relevant_features = 1
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    while cumsum[nb_relevant_features-1] < 0.8:
        nb_relevant_features += 1

    fig, axs = plt.subplots(2,1, sharex=True)

    axs[0].plot(list(range(1, len(pca.explained_variance_)+1)), pca.explained_variance_)
    axs[0].set_ylabel("Explained Variance")
    axs[0].grid()

    axs[1].plot(list(range(1, len(pca.explained_variance_)+1)), np.cumsum(pca.explained_variance_ratio_))
    axs[1].set_xlabel("Number of features")
    axs[1].set_ylabel("Cumulative Inertia Percentage")
    axs[1].plot(list(range(1, len(pca.explained_variance_)+1)), [0.8]*len(pca_features), color = "r", linestyle='dashed')
    axs[1].grid()

    fig.suptitle("Relevant feature analysis for PCA")

    plt.show()

    df_principal_components = df_principal_components[:,:nb_relevant_features]

    nb_plots = nb_relevant_features*(nb_relevant_features-1)//2
    n_cols = nb_plots//2
    n_rows = int(np.ceil(nb_plots/n_cols))
    n_cols = int(np.ceil(nb_plots/n_rows))      #If the number of plots is not even, find a better repartition if possible. Example: 15 -> (7, 3) -> (5, 3)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    counter = 0
    for featurey in range(nb_relevant_features):
        for featurex in range(featurey+1,nb_relevant_features):
            axs[counter//n_cols,counter%n_cols].scatter(df_principal_components[:,featurex], df_principal_components[:,featurey], c = Df[category] if category else None)
            axs[counter//n_cols,counter%n_cols].set_xlabel(f"Feature {featurex}")
            axs[counter//n_cols,counter%n_cols].set_ylabel(f"Feature {featurey}")
            counter += 1
    fig.suptitle("PCA Analysis")
    plt.show()

    eye_matrix = np.eye(len(pca_features))
    contribution_vectors = pca.transform(eye_matrix)[:,:nb_relevant_features]

    norms = {pca_features[vector]: np.linalg.norm(contribution_vectors[vector]) for vector in range(len(pca_features))}

    features_importance = sorted(norms.keys(), key=lambda item: norms[item], reverse=True)

    sim_matrix = cosine_similarity(contribution_vectors) - np.eye(len(pca_features))

    sim_pairs = []
    for i in range(len(pca_features)):
        for j in range(i):
            if sim_matrix[i,j] >= 0.8:
                sim_pairs.append([pca_features[i], pca_features[j]])

    return pca, nb_relevant_features, features_importance, sim_pairs