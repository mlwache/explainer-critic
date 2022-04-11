from typing import Tuple

import torch
from matplotlib.figure import Figure, Axes
from torch import Tensor
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation_experiments_console import set_up_evaluation_experiments
from utils import FastMNIST


def show_umap_plot():
    sns.set(context="paper", style="white")
    print("Setting up experiments...")
    explainers, test_loader, model_paths = set_up_evaluation_experiments(3, n_test_samples=10000)
    for i in range(len(explainers)):
        explanations, labels = explainers[i].get_labeled_explanations(test_loader, "input_x_gradient")
        explanations = explanations.detach()
        print(f"Embedding for model {model_paths[i]}:")
        explanation_figure, exp_ax = get_umap_figure(explanations, labels)
        exp_ax.set_title(f"Input x Gradient UMAP-Embedding of the model {model_paths[i]}", fontsize=18)
        explanation_figure.show()

    print("Embedding for the input:")
    mnist = FastMNIST('./data', train=False, download=True)
    input_figure, input_ax = get_umap_figure(mnist.data, mnist.targets)
    input_ax.set_title("MNIST inputs embedded into two dimensions by UMAP", fontsize=18)
    input_figure.show()


def get_umap_figure(data: Tensor, labels: Tensor) -> Tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(12, 10))
    reducer = umap.UMAP(random_state=42)
    data: Tensor = torch.flatten(data, start_dim=1, end_dim=3)

    print('computing embedding...')
    embedding = reducer.fit_transform(data)
    print('finished computing embedding')

    color = labels
    ax.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=5)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


if __name__ == '__main__':
    show_umap_plot()
