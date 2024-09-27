import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
np.set_printoptions(threshold=sys.maxsize)

class Soudure():
    def __init__(self, file):
        self.titles = ["Carbon concentration / (weight%)", "Silicon concentration / (weight%)", "Manganese concentration / (weight%)", "Sulphur concentration / (weight%)", "Phosphorus concentration / (weight%)", "Nickel concentration / (weight%)", "Chromium concentration / (weight%)", "Molybdenum concentration / (weight%)", "Vanadium concentration / (weight%)", "Copper concentration / (weight%)", "Cobalt concentration / (weight%)", "Tungsten concentration / (weight%)", "Oxygen concentration / parts per million by weight", "Titanium concentration / parts per million by weight", "Nitrogen concentration / parts per million by weight", "Aluminium concentration / parts per million by weight", "Boron concentration / parts per million by weight", "Niobium concentration / parts per million by weight", "Tin concentration / parts per million by weight", "Arsenic concentration / parts per million by weight", "Antimony concentration / parts per million by weight", "Current / A", "Voltage / V", "AC or DC", "Electrode positive or negative", "Heat input / kJmm-1", "Interpass temperature / °C", "Type of weld", "Post weld heat treatment temperature / °C", "Post weld heat treatment time / hours", "Yield strength / MPa", "Ultimate tensile strength / MPa", "Elongation / %", "Reduction of Area / %", "Charpy temperature / °C", "Charpy impact toughness / J", "Hardness / kgmm-2", "50% FATT", "Primary ferrite in microstructure / %", "Ferrite with second phase / %", "Acicular ferrite / %", "Martensite / %", "Ferrite with carbide aggreagate / %", "Weld ID"]
        self.create_database(file, delimiter=" ")

    def __len__(self):
        return len(self.Db)
    
    def create_database(self, file:str, delimiter:str):
        '''
        This function Creates the early preprocessed Pandas Dataframe corresponding to the desiganted file.
        
        Inputs: 
            - file (str):       Absolute or relative path to the Database sourcefile
            - delimiter (str):  Delimiter used in the Database sourcefile
        
        Output: None
        '''

        if type(delimiter) != str:
            raise ValueError("Delimiter type must be a string type")
        
        self.Db = pd.read_csv(file, delimiter = "\s+", names= self.titles)

        self.Db = self.Db.replace("N", np.nan)          #"N" values mean no measure. We replace this value with a more significant NaN name for ulterior treatement
        
        self.Db = self.Db.replace(to_replace=r'^<', value='0', regex=True)
        self.Db = self.Db.apply(pd.to_numeric, errors='ignore')
        
    def plot_histogram(self, channel):
        '''This function displays the histogram for the non-NaN values of the designated channel.
        
        Input: channel (str OR int): the name or the id of the channel to plot
        
        Output: None'''

        if type(channel) == int:
            channel = self.titles[channel]

        vector = np.array(self.Db[channel])
        print(vector)
        vector = vector[~np.isnan(vector)]      #Remove the NaN values
        
        if len(vector) == 0:
            raise ValueError("Not enough values to create a histogram")
        
        sns.histplot(vector, kde = True)
        
        plt.show()



if __name__ == "__main__":
    S = Soudure("welddb\welddb.data")
    S.plot_histogram(9)