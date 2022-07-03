import numpy as np
import pandas as pd


def reduce_memory_usage(dataframe: pd.DataFrame, verbose: bool = True):

    start_memory = dataframe.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {start_memory} MB")

    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        if col_type != "object":
            c_min = dataframe[col].min()
            c_max = dataframe[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)

            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    pass
        else:
            dataframe[col] = dataframe[col].astype("category")

    if verbose:
        end_memory = dataframe.memory_usage().sum() / 1024**2
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")

    return dataframe
