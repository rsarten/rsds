from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..utilities.transform.core import scale_data


def generate_pca_loadings(data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Generate PCA values and return feature loadings for each PCA
    """
    pca = PCA()
    data_pca = pca.fit_transform(data)
    component_names = [f"PC{i+1}" for i in range(data_pca.shape[1])]

    loadings = pd.DataFrame(
        pca.components_.T, columns=component_names, index=data.columns  # type: ignore
    )
    expl_var = pca.explained_variance_ratio_  # type: ignore

    return loadings, expl_var


def plot_pca_loadings(
    loadings: pd.DataFrame,
    expl_var: np.ndarray,
    components: int,
    show: bool = True,
    save_file: Optional[str] = None,
) -> None:
    labels = [f"Exp Var {(r*100):.2f}%" for r in expl_var[0:components]]
    fsize = loadings.shape[0]
    title = {"weight": "bold"}
    fig, ax = plt.subplots(1, components, figsize=(20, fsize // 2))

    ax[0].barh(loadings.index, loadings.iloc[:, 0])
    ax[0].set_title(labels[0], title)

    for i in range(1, components):
        ax[i].barh(loadings.index, loadings.iloc[:, i])
        ax[i].axes.get_yaxis().set_visible(False)
        ax[i].set_title(labels[i], title)

    if save_file is not None:
        plt.savefig(save_file)
    if not show:
        plt.close(fig)
    else:
        plt.show()


def pca_loadings(
    data: pd.DataFrame,
    components: int = 5,
    show: bool = True,
    save_file: Optional[str] = None,
) -> None:
    scaled = scale_data(data)
    scaled.fillna(0, inplace=True)
    loadings, expl_var = generate_pca_loadings(scaled)
    plot_pca_loadings(loadings, expl_var, components, show, save_file)
