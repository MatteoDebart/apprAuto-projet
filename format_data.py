import pandas as pd
from utils import convert_less_than


COLUMNS = ["Carbon concentration / (weight%)", "Silicon concentration / (weight%)", "Manganese concentration / (weight%)", "Sulphur concentration / (weight%)", "Phosphorus concentration / (weight%)", "Nickel concentration / (weight%)", "Chromium concentration / (weight%)", "Molybdenum concentration / (weight%)", "Vanadium concentration / (weight%)", "Copper concentration / (weight%)", "Cobalt concentration / (weight%)", "Tungsten concentration / (weight%)", "Oxygen concentration / parts per million by weight", "Titanium concentration / parts per million by weight", "Nitrogen concentration / parts per million by weight", "Aluminium concentration / parts per million by weight", "Boron concentration / parts per million by weight", "Niobium concentration / parts per million by weight", "Tin concentration / parts per million by weight", "Arsenic concentration / parts per million by weight", "Antimony concentration / parts per million by weight", "Current / A", "Voltage / V", "AC or DC", "Electrode positive or negative", "Heat input / kJmm-1", "Interpass temperature / °C", "Type of weld", "Post weld heat treatment temperature / °C", "Post weld heat treatment time / hours", "Yield strength / MPa", "Ultimate tensile strength / MPa", "Elongation / %", "Reduction of Area / %", "Charpy temperature / °C", "Charpy impact toughness / J", "Hardness / kgmm-2", "50% FATT", "Primary ferrite in microstructure / %", "Ferrite with second phase / %", "Acicular ferrite / %", "Martensite / %", "Ferrite with carbide aggreagate / %", "Weld ID"]

CATEGORICAL_COL = ["AC or DC", "Electrode positive or negative", "Type of weld", "Weld ID"]

NUMERICAL_COL = list(set(COLUMNS) - set(CATEGORICAL_COL))


def create_dataframe(file_path):

    db = pd.read_csv(file_path, delimiter = "\s+", names=COLUMNS, na_values='N')

    db[NUMERICAL_COL] = db[NUMERICAL_COL].applymap(convert_less_than)
    db[NUMERICAL_COL] = db[NUMERICAL_COL].apply(pd.to_numeric, errors='coerce')
    
    db.to_csv("table.csv", index=False)

    return db

if __name__=='__main__':

    file_path="welddb/welddb.data"
    create_dataframe(file_path)