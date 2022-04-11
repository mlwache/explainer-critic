from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
from matplotlib.figure import Figure, Axes
from torch import Tensor

import utils
from evaluation_experiments_console import set_up_evaluation_experiments


def show_plots():
    sns.set(context="paper", style="white")
    fontsize_pca = 12
    fontsize_umap = 22
    print("Setting up experiments...")
    explainers, test_loader, model_paths = set_up_evaluation_experiments(3, n_test_samples=10000)

    for i in range(len(explainers)):
        explanations, labels = explainers[i].get_labeled_explanations(test_loader, "input_x_gradient")
        explanations = explanations.detach()

        # PCA on explanations
        print(f"PCA embedding for model {model_paths[i]}:")
        print(f"largest variance component: {pca_largest_variance(explanations)}")

        explanation_figure_pca, exp_ax_pca = get_pca_figure(explanations, labels)
        exp_ax_pca.set_title(f"Input x Gradient PCA-Embedding of the model {model_paths[i]}", fontsize=fontsize_pca)
        explanation_figure_pca.show()

        # UMAP on explanations
        print(f"UMAP embedding for model {model_paths[i]}:")
        explanation_figure_umap, exp_ax_umap = get_umap_figure(explanations, labels)
        exp_ax_umap.set_title(f"Input x Gradient UMAP-Embedding of the model {model_paths[i]}", fontsize=fontsize_umap)
        explanation_figure_umap.show()

    print("Embedding for the input:")
    inputs, labels = utils.get_data_tensors(n_samples=10000)

    # PCA for inputs
    print(f"largest variance component: {pca_largest_variance(inputs)}")

    input_figure_pca, input_ax_pca = get_pca_figure(inputs, labels)
    input_ax_pca.set_title("Input x Gradient PCA-Embedding of the inputs", fontsize=fontsize_pca)
    input_figure_pca.show()

    # UMAP for inputs
    input_figure_umap, input_ax_umap = get_umap_figure(inputs, labels)
    input_ax_umap.set_title("Input x Gradient UMAP-Embedding of the inputs", fontsize=fontsize_umap)
    input_figure_umap.show()


def get_umap_figure(data: Tensor, labels: Tensor) -> Tuple[Figure, Axes]:
    data: Tensor = torch.flatten(data, start_dim=1, end_dim=3)

    reducer = umap.UMAP(random_state=42)
    print('computing embedding...')
    embedding = reducer.fit_transform(data)
    print('finished computing embedding')

    fig, ax = plt.subplots(figsize=(12, 10))
    color = labels
    ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=5)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def _pca_projection_and_variance(inputs: Tensor) -> Tuple[float, Tensor]:
    inputs = torch.flatten(inputs, start_dim=1, end_dim=3)
    u, s, v = torch.pca_lowrank(inputs, q=100)
    projected = inputs @ v[:, :2]
    return s[0], projected


def pca_2d_projection(inputs: Tensor) -> Tensor:
    return _pca_projection_and_variance(inputs)[1]


def pca_largest_variance(inputs: Tensor) -> float:
    return _pca_projection_and_variance(inputs)[0]


def get_pca_figure(inputs: Tensor, labels: Tensor) -> Tuple[Figure, Axes]:
    projected = pca_2d_projection(inputs)
    fig, ax = plt.subplots()
    ax.scatter(projected.T[0], projected.T[1], c=labels, s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


if __name__ == '__main__':
    show_plots()
