from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .vis_utils import check_col_present, figax, move_frame

# Ignoring Typing issues for accessing plt.Axes elements and numpy types


def plot_hist(
    row: int, col: int, axes: Axes, data: pd.DataFrame, col_name: str, **kwargs
) -> None:
    axes[row, col].hist(data[col_name], **kwargs)  # type: ignore


def plot_scat(
    row: int, col: int, axes: Axes, data: pd.DataFrame, colx: str, coly: str, **kwargs
) -> None:
    axes[row, col].scatter(data[colx], data[coly], **kwargs)  # type: ignore


def build_plots(data: pd.DataFrame, col_n: int, f_plot: Callable = plot_hist) -> Figure:
    # plt.rcParams["figure.figsize"] = (25, 20)
    fig, axes = figax(data, col_n)
    row, col = (0, 0)

    for col_name in data.columns:
        # ax[j, i].hist(data[col], bins = 100)
        f_plot(row, col, axes, data, col_name)
        axes[row, col].set_title(col_name, {"weight": "bold"})  # type: ignore
        row, col = move_frame(row, col, col_n)

    return fig


def plot_numerics(
    data: pd.DataFrame,
    col_n: int,
    target: Optional[str] = None,
    bins: int = 100,
    show: bool = True,
    save_file: Optional[str] = None,
) -> None:
    """
    Plot grid of numeric features. If target is not supplied, will plot
    histograms, else will plot scatter against target.
    """

    numeric_data = data.select_dtypes(include=np.number)  # type: ignore

    if target is None:
        plot_ = partial(plot_hist, bins=bins)
    else:
        check_col_present(target, numeric_data)
        plot_ = partial(plot_scat, coly=target)

    fig = build_plots(numeric_data, col_n, f_plot=plot_)
    fig.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    if not show:
        plt.close(fig)
    else:
        plt.show()


def corr_numerics(
    data: pd.DataFrame,
    target: Optional[str] = None,
    show: bool = True,
    save_file: Optional[str] = None,
) -> None:
    """
    Generate visualisation of correlation between numeric features.

    If 'target' is None, displays full nxn grid. If target is supplied then
    only displays 1xn for row of target.
    """

    numeric_data: pd.DataFrame = data.select_dtypes(include=np.number)  # type: ignore
    sizex: int = numeric_data.shape[1]
    sizey: int = sizex
    corr_vals: pd.DataFrame = numeric_data.corr()

    if target is not None:
        check_col_present(target, numeric_data)
        sizey = 2
        target_idx = numeric_data.columns.get_loc(target)
        corr_vals = corr_vals.iloc[target_idx : target_idx + 1]

    plt.figure(figsize=(sizex, sizey))
    axes = sns.heatmap(corr_vals, cmap="viridis", annot=True)
    fig = axes.get_figure()

    if save_file is not None:
        fig.savefig(save_file)
    if not show:
        plt.close(fig)
    else:
        plt.show()
