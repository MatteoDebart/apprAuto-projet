import numpy as np
import pandas as pd

NUMERICAL_COL = [
    "Carbon concentration / (weight%)", 
    "Silicon concentration / (weight%)", 
    "Manganese concentration / (weight%)",
    "Sulphur concentration / (weight%)", 
    "Phosphorus concentration / (weight%)", 
    "Nickel concentration / (weight%)",
    "Chromium concentration / (weight%)", 
    "Molybdenum concentration / (weight%)", 
    "Vanadium concentration / (weight%)",
    "Copper concentration / (weight%)", 
    "Cobalt concentration / (weight%)", 
    "Tungsten concentration / (weight%)",
    "Oxygen concentration / parts per million by weight", 
    "Titanium concentration / parts per million by weight",
    "Nitrogen concentration / parts per million by weight", 
    "Aluminium concentration / parts per million by weight",
    "Boron concentration / parts per million by weight", 
    "Niobium concentration / parts per million by weight",
    "Tin concentration / parts per million by weight", 
    "Arsenic concentration / parts per million by weight",
    "Antimony concentration / parts per million by weight", 
    "Current / A", 
    "Voltage / V", 
    "Heat input / kJmm-1", 
    "Interpass temperature / °C",
    "Post weld heat treatment temperature / °C", 
    "Post weld heat treatment time / hours", 
    "Yield strength / MPa", 
    "Ultimate tensile strength / MPa",
    "Elongation / %", 
    "Reduction of Area / %", 
    "Charpy temperature / °C", 
    "Charpy impact toughness / J", 
    "Hardness / kgmm-2",
    "Primary ferrite in microstructure / %", 
    "Ferrite with second phase / %", 
    "Acicular ferrite / %", 
    "Martensite / %", 
    "Ferrite with carbide aggreagate / %"
]

CATEGORICAL_COL = [
    "AC or DC", 
    "Electrode positive or negative", 
    "Type of weld", 
    "50% FATT", 
    "Weld ID"
]

COLUMNS = NUMERICAL_COL+CATEGORICAL_COL

def convert_less_than(value):
    if isinstance(value, str) and value.startswith('<'):
        try:
            number = float(value[1:])
            return np.random.uniform(0, number)
        except ValueError:
            return value
    return value

def create_dataframe(file_path):
    Db = pd.read_csv(file_path, delimiter = "\s+", names=COLUMNS, na_values='N')
    Db[NUMERICAL_COL] = Db[NUMERICAL_COL].applymap(convert_less_than)
    Db[NUMERICAL_COL] = Db[NUMERICAL_COL].apply(pd.to_numeric, errors='coerce')

    Db.to_csv("table.csv")
    return Db

if __name__=='__main__':
    file_path="welddb/welddb.data"
    create_dataframe(file_path)