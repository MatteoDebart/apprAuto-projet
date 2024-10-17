import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
np.set_printoptions(threshold=sys.maxsize)

def convert_less_than(value):
    if isinstance(value, str) and value.startswith('<'):
        try:
            number = float(value[1:])
            return np.random.uniform(0, number)
        except ValueError:
            return value
    return value

class Soudure():
    def __init__(self, file):
        self.COLUMNS = ['Current / A',
            'Heat input / kJmm-1',
            'Post weld heat treatment temperature / °C',
            'Niobium concentration / parts per million by weight',
            'Molybdenum concentration / (weight%)',
            'Type of weld_GMAA',
            'Vanadium concentration / (weight%)',
            'Type of weld_MMA',
            'Type of weld_SA',
            'Nitrogen concentration / parts per million by weight',
            'Phosphorus concentration / (weight%)',
            'Type of weld_SAA',
            'Carbon concentration / (weight%)',
            'Manganese concentration / (weight%)',
            'Voltage / V',
            'Chromium concentration / (weight%)',
            'AC or DC',
            'Type of weld_GTAA',
            'Sulphur concentration / (weight%)']

        self.create_dataframe(file)

    def __len__(self):
        return len(self.Db)
    
    def create_dataframe(self, file_path:str):
        '''
        This function Creates the early preprocessed Pandas Dataframe corresponding to the desiganted file.
        
        Inputs: 
            - file (str):       Absolute or relative path to the Database sourcefile
        
        Output: None
        '''

        if os.path.isfile(file_path):
            self.Db = pd.read_excel(file_path, names=self.COLUMNS)
            self.MNAR_Analysis()
            self.MAR_Analysis()
        
        else:
            raise ValueError("Path not found")

        
    def MNAR_Analysis(self):
        self.Missing_mask = self.Db.isna().replace({False: 0, True: 1})

        corellation_matrix = np.abs(np.corrcoef(np.random.normal(np.array(self.Missing_mask), 1e-9), rowvar=False) - np.eye(len(self.COLUMNS),len(self.COLUMNS)))
        np.savetxt("Correlation_matrix.csv", corellation_matrix, delimiter = ";")

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

    def MAR_Analysis(self):
        #self.Missing_mask = self.Db.isna().replace({False: 0, True: 1})
        self.Missing_mask = self.Db.isna()
        for i in self.COLUMNS:
            for j in self.COLUMNS:
                fig, axs = plt.subplots(3, 1, sharex=True)
                fig.suptitle(i)

                vector = np.array(self.Db[j])
                vector = vector[~np.isnan(vector)]      #Remove the NaN values
                _ , bins = np.histogram(vector, 100)
                axs[0].hist(vector, bins=bins, density = True, label = "With Missing Values")
                axs[0].legend()

                vector2 = np.array(self.Db[j].where(~self.Missing_mask[i]))
                vector2 = vector2[~np.isnan(vector2)]      #Remove the NaN values
                vector3 = np.array(self.Db[j].where(self.Missing_mask[i]))
                vector3 = vector3[~np.isnan(vector3)]      #Remove the NaN values

                if np.sum(np.abs(vector2)) != 0 and vector2.shape[0] <= 0.8*vector.shape[0]:
                    print(vector.shape)
                    print(vector2.shape)
                    print(vector3.shape)
                    axs[1].hist(vector2,bins=bins, density = True, label = "Without Missing Values")
                    axs[1].legend()
                    axs[2].hist(vector3, bins=bins, density = True, label = "Missing Values Distribution")
                    axs[2].legend()
                    plt.xlabel(j)
                    plt.legend()
                    plt.show()

                else:
                    print(i, j)
                    plt.close(fig)


if __name__ == "__main__":
    S = Soudure("relvant_features.xlsx")