import numpy as np
import pandas as pd


def reduce_memory_usage(dataframe: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:

    start_memory = dataframe.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {start_memory:.2f} MB")

    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        if col_type != "object":
            if str(col_type)[:3] == "int":
                dataframe[col] = mod_type_int(dataframe[col])
            else:
                dataframe[col] = mod_type_float(dataframe[col])
        else:
            dataframe[col] = dataframe[col].astype("category")

    if verbose:
        end_memory = dataframe.memory_usage().sum() / 1024**2
        print(f"Memory usage of dataframe after reduction {end_memory:.2f} MB")
        print(f"Reduced by {(100 * (start_memory - end_memory) / start_memory):.2f} % ")

    return dataframe


def mod_type_int(col: pd.Series) -> pd.Series:
    """Modify Series if integer type"""
    c_min = col.min()
    c_max = col.max()

    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        col = col.astype(np.int8)
    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        col = col.astype(np.int16)
    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        col = col.astype(np.int32)
    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        col = col.astype(np.int64)

    return col


def mod_type_float(col: pd.Series) -> pd.Series:
    """Modify Series if float type"""
    c_min = col.min()
    c_max = col.max()

    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        col = col.astype(np.float16)
    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
        col = col.astype(np.float32)

    return col
