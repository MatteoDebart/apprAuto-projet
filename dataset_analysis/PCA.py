import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from preprocess import *
import itertools

def preprocess_PCA(Db:pd.DataFrame):
    # Outliers and scaling
    Db=handle_outliers(Db)
    scaler = StandardScaler()
    scaled_feature = get_numerical_features(Db)
    Db[scaled_feature] = scaler.fit_transform(Db[scaled_feature])

    Db = imputation(Db, categorical=False)
    features = list(set(get_numerical_features(Db))-set(MECHANICAL_PROPERTIES))
    Db = Db[features]
    return Db

def run_PCA(Db:pd.DataFrame):
    pca = PCA(n_components=len(Db.columns))
    principal_components = pca.fit_transform(Db)

    nb_relevant_features = 1
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    while cumsum[nb_relevant_features-1] < 0.8:
        nb_relevant_features += 1
    return pca, principal_components, nb_relevant_features

def plot_explained_variance_pca(pca:PCA, nb_relevant_features):
    '''
    Plots the explained variance.

    Inputs:
        - pca: The fitted PCA object.
        - nb_relevant_features: Number of relevant components to plot.
    '''
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = len(pca.explained_variance_ratio_)
    
    # Plot all components with color distinction
    plt.figure(figsize=(10, 6))
    
    # Plot relevant components in one color
    plt.plot(range(1, nb_relevant_features+1), 
             explained_variance[:nb_relevant_features], 
             marker='o', linestyle='-', color='b', label='Relevant Components')
    
    # Plot non-relevant components in a different color
    plt.plot(range(nb_relevant_features+1, num_components+1), 
             explained_variance[nb_relevant_features:], 
             marker='o', linestyle='-', color='gray', label='Other Components')
    
    # Add labels and title
    plt.axhline(0.8, color='r', linestyle='--', label="80% Explained Variance Threshold")
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.show()

def plot_pca_distribution(principal_components, ax, component_x:int, component_y: int, component_z: Optional[int] = None):
    '''
    Plots the distribution of data points in the PCA-reduced space.
    
    Inputs:
        - principal_components: The transformed data from PCA.
        - nb_relevant_features: Number of relevant components.
        - ax: Matplotlib axis for plotting.
        - component_x: Index of the first component to plot (1-based indexing).
        - component_y: Index of the second component to plot (1-based indexing).
        - component_z: Optional index of the third component to plot (for 3D plotting).
    '''

    if component_z is None:
        # 2D Scatter plot for 2 principal components
        ax.scatter(principal_components[:, component_x], principal_components[:, component_y], alpha=0.7)
        ax.set_xlabel(f"PC{component_x}")
        ax.set_ylabel(f"PC{component_y}")
        ax.set_title(f"PCA Distribution (PC{component_x} vs PC{component_y})")
    else:
        # 3D Scatter plot for 3 principal 
        ax = plt.axes(projection='3d')
        ax.scatter3D(principal_components[:, component_x], principal_components[:, component_y], principal_components[:, component_z], alpha=0.7)
        ax.set_xlabel(f"PC{component_x}")
        ax.set_ylabel(f"PC{component_y}")
        ax.set_zlabel(f"PC{component_z}")
        ax.set_title(f"PCA Distribution (PC{component_x}, PC{component_y}, PC{component_z})")

def plot_correlation_circle(pca:PCA, columns, ax, component_x, component_y, threshold=0.05):
    '''
    Plots the PCA correlation circle for two principal components.
    
    Inputs:
        - pca: Fitted PCA object.
        - columns: List of original feature names from the DataFrame.
        - ax: Matplotlib axis for plotting.
        - component_x: Index of the first principal component (1-based indexing).
        - component_y: Index of the second principal component (1-based indexing).
    '''
    
    # Get the loadings (correlations with the principal components)
    loadings = pca.components_[[component_x, component_y]]
    # Plot settings
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Draw a unit circle
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Plot each feature as a vector
    for i, feature in enumerate(columns):
        var = loadings[0, i]**2 + loadings[1, i]**2
        if var<threshold:
            continue
        ax.arrow(0, 0, loadings[0, i], loadings[1, i], 
                 color='b', alpha=0.5, head_width=0.05, head_length=0.05)
        ax.text(loadings[0, i] * 1.1, loadings[1, i] * 1.1, feature, 
                color='g', ha='center', va='center')

    # Set axis labels
    ax.set_xlabel(f"PC{component_x} ({pca.explained_variance_ratio_[component_x]*100:.2f}% variance)")
    ax.set_ylabel(f"PC{component_y} ({pca.explained_variance_ratio_[component_y]*100:.2f}% variance)")
    
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title(f"Correlation Circle (PC{component_x} vs PC{component_y})")
    ax.grid(True)


def plot_PCA(pca: PCA, principal_components, nb_relevant_features: int, columns: list[str]):
    '''
    For each pair of principal components under the top nb_relevant_features,
    this function plots the PCA distribution and PCA correlation circle side by side.
    
    Inputs:
        - pca: The fitted PCA object.
        - principal_components: The PCA-transformed data (output of pca.fit_transform).
        - nb_relevant_features: Number of relevant components to visualize.
        - columns: List of original feature names from the DataFrame.
    '''
    # Generate all pairs of components within the relevant features
    pairs = list(itertools.combinations(range(nb_relevant_features), 2))

    # Sort the pairs by the sum of the explained variance of the two components, in descending order
    pairs.sort(key=lambda x: pca.explained_variance_ratio_[x[0]] + pca.explained_variance_ratio_[x[1]], reverse=True)

    # Plot each pair
    for (i, j) in pairs:
        # Create a new figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot the PCA distribution for the current pair of components (i, j)
        plot_pca_distribution(principal_components, ax1, component_x=i, component_y=j, component_z=None)
        
        # Plot the PCA correlation circle for the current pair of components (i, j)
        plot_correlation_circle(pca, columns, ax2, component_x=i, component_y=j)
        
        # Adjust the layout to avoid overlapping labels
        plt.tight_layout()

        # Show the plot
        plt.show()

if __name__=='__main__':
    Db=pd.read_excel('table.xlsx')
    pca_datset=preprocess_PCA(Db)
    pca, principal_components, nb_relevant_features = run_PCA(pca_datset)
    plot_PCA(pca, principal_components, nb_relevant_features, pca_datset.columns)
    plot_explained_variance_pca(pca, nb_relevant_features)