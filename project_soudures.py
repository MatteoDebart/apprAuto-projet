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
        self.COLUMNS = ["Carbon concentration / (weight%)", "Silicon concentration / (weight%)", "Manganese concentration / (weight%)", "Sulphur concentration / (weight%)", "Phosphorus concentration / (weight%)", "Nickel concentration / (weight%)", "Chromium concentration / (weight%)", "Molybdenum concentration / (weight%)", "Vanadium concentration / (weight%)", "Copper concentration / (weight%)", "Cobalt concentration / (weight%)", "Tungsten concentration / (weight%)", "Oxygen concentration / parts per million by weight", "Titanium concentration / parts per million by weight", "Nitrogen concentration / parts per million by weight", "Aluminium concentration / parts per million by weight", "Boron concentration / parts per million by weight", "Niobium concentration / parts per million by weight", "Tin concentration / parts per million by weight", "Arsenic concentration / parts per million by weight", "Antimony concentration / parts per million by weight", "Current / A", "Voltage / V", "AC or DC", "Electrode positive or negative", "Heat input / kJmm-1", "Interpass temperature / °C", "Type of weld", "Post weld heat treatment temperature / °C", "Post weld heat treatment time / hours", "Yield strength / MPa", "Ultimate tensile strength / MPa", "Elongation / %", "Reduction of Area / %", "Charpy temperature / °C", "Charpy impact toughness / J", "Hardness / kgmm-2", "50% FATT", "Primary ferrite in microstructure / %", "Ferrite with second phase / %", "Acicular ferrite / %", "Martensite / %", "Ferrite with carbide aggreagate / %", "Weld ID"]

        self.CATEGORICAL_COL = [
            "AC or DC", 
            "Electrode positive or negative", 
            "Type of weld",  
            "Weld ID"
]
        self.NUMERICAL_COL = list(set(self.COLUMNS) - set(self.CATEGORICAL_COL))

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
            self.Db = pd.read_csv(file_path, delimiter = "\s+", names=self.COLUMNS, na_values='N')
            self.Db[self.NUMERICAL_COL] = self.Db[self.NUMERICAL_COL].applymap(convert_less_than)
            self.Db[self.NUMERICAL_COL] = self.Db[self.NUMERICAL_COL].apply(pd.to_numeric, errors='coerce')

            self.Db.to_csv("table.csv")
            self.plot_histogram(2)
            self.MNAR_Analysis()
            self.MAR_Analysis()
        
        else:
            raise ValueError("Path not found")
        
    def plot_histogram(self, channel):
        '''This function displays the histogram for the non-NaN values of the designated channel.
        
        Input: channel (str OR int): the name or the id of the channel to plot
        
        Output: None'''

        if type(channel) == int:
            channel = self.COLUMNS[channel]

        if channel in self.NUMERICAL_COL:
            vector = np.array(self.Db[channel])
            vector = vector[~np.isnan(vector)]      #Remove the NaN values
            
            if len(vector) == 0:
                raise ValueError("Not enough values to create a histogram")
            
            sns.histplot(vector, kde = True)
            
            plt.show()

        else:
            order = self.Db[channel].value_counts().index
            vector = np.array(self.Db[channel])
            vector = vector[~np.isnan(vector)]      #Remove the NaN values
            
            if len(vector) == 0:
                raise ValueError("Not enough values to create a histogram")
            
            sns.countplot(y=vector, order=order)
            
            plt.show()
        
    def MNAR_Analysis(self):
        self.Missing_mask = self.Db.isna().replace({False: 0, True: 1})

        corellation_matrix = np.abs(np.corrcoef(np.random.normal(np.array(self.Missing_mask)[:,:-1], 1e-9), rowvar=False) - np.eye(len(self.COLUMNS)-1,len(self.COLUMNS)-1))
        np.savetxt("Correlation_matrix.csv", corellation_matrix, delimiter = ";")

        mask = np.array([[x>=y for y in range(len(self.COLUMNS)-1)] for x in range(len(self.COLUMNS)-1)])
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
            for j in self.NUMERICAL_COL:
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
    S = Soudure("welddb\welddb.data")