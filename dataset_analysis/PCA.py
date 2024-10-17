import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def plot_PCA(Df:pd.DataFrame, pca_features:list, category:str = None):
    '''
    This function realises the PCA Analysis, decides how many features are relevant using the 80% of cumulative inertia property and plots the 2D graph of each pair of relevant feature.


    Input:  - Df:               The Dataframe containing the Dataset with imputed missing values
            - pca_features:     The names of the columns considered as input of the PCA
            - category:         The name of the column of the categorial feture used for colors in the final plot
    
    Returns: pca:                   The instance of PCA trained on the features
             nb_relevant_features   The number of relevant PCA features
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
            axs[counter//n_cols,counter%n_cols].scatter(df_principal_components[:,featurex], df_principal_components[:,featurey], c = Df[category])
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