import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def num_rows_cols(data: pd.DataFrame, cols: int = 5) -> tuple[int, int]:
    """
    Return grid dimensions given specified number of columns, and number of
    columns in data that are to be displayed.
    """
    num_cols = data.shape[1]
    num_rows = int(np.ceil(num_cols / cols))
    cols = min(cols, num_cols)
    return num_rows, cols


def figax(data: pd.DataFrame, col_n: int = 5) -> tuple[Figure, list[list[Axes]]]:
    """
    Return subplots (Figuere, Axes) for given data with specified column width.
    """
    n_rows, n_cols = num_rows_cols(data, col_n)
    return plt.subplots(n_rows, n_cols, figsize=(36, 24))


def move_frame(row: int, col: int, cols: int) -> tuple[int, int]:
    """
    Step through Axes matrix within column number bounds.
    """
    if col == (cols - 1):
        col = 0
        row += 1
    else:
        col += 1
    return row, col


def check_col_present(target: str, data: pd.DataFrame) -> None:
    if target not in data.columns:
        raise LookupError(f"{target} is not member of numeric features")
