from typing import Optional
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

COLUMNS = ["Carbon concentration / (weight%)", "Silicon concentration / (weight%)", "Manganese concentration / (weight%)", "Sulphur concentration / (weight%)", "Phosphorus concentration / (weight%)", "Nickel concentration / (weight%)", "Chromium concentration / (weight%)", "Molybdenum concentration / (weight%)", "Vanadium concentration / (weight%)", "Copper concentration / (weight%)", "Cobalt concentration / (weight%)", "Tungsten concentration / (weight%)", "Oxygen concentration / parts per million by weight", "Titanium concentration / parts per million by weight", "Nitrogen concentration / parts per million by weight", "Aluminium concentration / parts per million by weight", "Boron concentration / parts per million by weight", "Niobium concentration / parts per million by weight",
           "Tin concentration / parts per million by weight", "Arsenic concentration / parts per million by weight", "Antimony concentration / parts per million by weight", "Current / A", "Voltage / V", "AC or DC", "Electrode positive or negative", "Heat input / kJmm-1", "Interpass temperature / °C", "Type of weld", "Post weld heat treatment temperature / °C", "Post weld heat treatment time / hours", "Yield strength / MPa", "Ultimate tensile strength / MPa", "Elongation / %", "Reduction of Area / %", "Charpy temperature / °C", "Charpy impact toughness / J", "Hardness / kgmm-2", "50% FATT", "Primary ferrite in microstructure / %", "Ferrite with second phase / %", "Acicular ferrite / %", "Martensite / %", "Ferrite with carbide aggreagate / %", "Weld ID"]


CATEGORICAL_COL = [
    "AC or DC",
    "Electrode positive or negative",
    "Type of weld",
    "Weld ID"
]

MECHANICAL_PROPERTIES = [
    "Yield strength / MPa",
    "Ultimate tensile strength / MPa",
    "Elongation / %",
    "Reduction of Area / %",
    "Charpy temperature / °C",
    "Charpy impact toughness / J",
    "Hardness / kgmm-2",
    "50% FATT",
    "Primary ferrite in microstructure / %",
    "Ferrite with second phase / %",
    "Acicular ferrite / %",
    "Martensite / %",
    "Ferrite with carbide aggreagate / %"
]

NUMERICAL_COL = list(set(COLUMNS) - set(CATEGORICAL_COL))


def AC_DC(x: Optional[str]):
    if x == 'AC':
        return 0
    if x == 'DC':
        return 1
    return None


def electrode(x: Optional[str]):
    if x == '0':
        return 0
    if x == '+':
        return 1
    if x == '-':
        return -1
    return None


def convert_less_than(value):
    if isinstance(value, str) and value.startswith('<'):
        try:
            number = float(value[1:])
            return np.random.uniform(0, number)
        except ValueError:
            return value
    return value


def create_dataframe(file_path: str = "welddb/welddb.data"):
    '''
    Create the dataframe and process a few of them
    '''
    Db = pd.read_csv(file_path, delimiter="\s+", names=COLUMNS, na_values='N')
    Db[NUMERICAL_COL] = Db[NUMERICAL_COL].map(convert_less_than)
    Db[NUMERICAL_COL] = Db[NUMERICAL_COL].apply(pd.to_numeric, errors='coerce')

    # Handling categorical
    Db["AC or DC"] = Db["AC or DC"].apply(AC_DC)
    Db["Electrode positive or negative"] = Db["Electrode positive or negative"].apply(
        electrode)

    encoder = OneHotEncoder(sparse_output=False)
    encoded_weld_type = encoder.fit_transform(Db[["Type of weld"]])
    encoded_weld_df = pd.DataFrame(
        encoded_weld_type, columns=encoder.get_feature_names_out(['Type of weld']))

    # Concatenate the encoded one-hot columns with the original dataframe
    Db = pd.concat([Db, encoded_weld_df], axis=1)

    Db = Db.drop('Weld ID', axis=1)
    Db = Db.drop('Type of weld', axis=1)
    return Db


if __name__ == '__main__':

    file_path = "welddb/welddb.data"
    create_dataframe(file_path)
