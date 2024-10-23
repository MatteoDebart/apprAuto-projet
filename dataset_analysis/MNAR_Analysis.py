import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))

from format_data import MECHANICAL_PROPERTIES, COLUMNS, create_dataframe
from utils import get_corr
from preprocess import feature_selection, OutputColumn
np.set_printoptions(threshold=sys.maxsize)

class MNAR_ANALYSIS():
    def __init__(self, file):
        self.create_df(file)

    def __len__(self):
        return len(self.Db)
    
    def create_df(self, file_path:str):
        '''
        This function Creates the early preprocessed Pandas Dataframe corresponding to the desiganted file.
        
        Inputs: 
            - file (str):       Absolute or relative path to the Database sourcefile
        
        Output: None
        '''

        if os.path.isfile(file_path):
            self.Db = create_dataframe(file_path)
            self.Db = self.Db.rename(columns={OutputColumn.yield_strength.value: 'output'})
            features = list(set(self.Db.columns)-set(MECHANICAL_PROPERTIES))
            col_info = get_corr(self.Db[features])
            features = feature_selection(col_info)
            self.Db = self.Db[features]
            self.COLUMNS = features
            self.MNAR_Analysis()
        
        else:
            raise ValueError("Path not found")

        
    def MNAR_Analysis(self):
        '''
        This function Analyses the correlation between the missingness of features. It creates a csv file containing the correlation matrix and plots the curve of
        Number of correlated pairs VS treshold. We add a very low-amplitude Gaussian noise to the one-hot encoding to prevent nan values for the constant columns.
        Since the noises are not correlated, this will likely not influence the Matrix in a significant way. 
        '''

        self.Missing_mask = self.Db.isna().replace({False: 0, True: 1})

        corellation_matrix = np.abs(np.corrcoef(np.random.normal(np.array(self.Missing_mask), 1e-9), rowvar=False) - np.eye(len(self.COLUMNS),len(self.COLUMNS)))

        corellation_df = pd.DataFrame(corellation_matrix, index = self.COLUMNS, columns=self.COLUMNS)
        corellation_df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Correlation_matrix.csv"), ";")

        mask = np.array([[x>=y for y in range(len(self.COLUMNS))] for x in range(len(self.COLUMNS))])
        X = np.linspace(0,1,100)
        Y = np.array([np.sum(mask*(corellation_matrix>=i))/np.sum(mask) for i in X])
        area = np.trapz(Y, dx = 0.01)
        plt.plot(X,Y, label = f"Area under Curve = {area:.4f}")
        plt.xlabel("correlation score treshold")
        plt.ylabel("proportion of pairs of variables considered as correlated")
        plt.title("MNAR Analysis")
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    S = MNAR_ANALYSIS("welddb/welddb.data")