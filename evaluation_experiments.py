from statistics import mean
from typing import Any, Tuple, List, Dict

import streamlit as st
import torch
from streamlit import session_state as state
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import utils
from config import SimpleArgumentParser
from explainer import Explainer


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained.

    Expects that the arguments are the same as for the model that should be evaluated"""

    modes = ["gradient", "input_x_gradient", "input"]
    if 'explainers' not in st.session_state:  # this part will run only once in the beginning.
        state.explainers, state.test_loader, state.visualization_loader, state.device = set_up_evaluation_experiments()

        # all of the following are lists, because I compute them for multiple models.
        state.accuracies = []
        state.intra_class_variances = []
        state.aggregated_variances = []

        for model_nr in range(2):
            intra_class_variances_per_model: Dict[str, List[float]] = {}
            aggregated_variance_per_model: Dict[str, float] = {}
            for mode in modes:
                # first, get  all explanations/inputs of the test set
                labeled_explanations: List[Tuple[Tensor, int]] = get_labeled_explanations(state.explainers[model_nr],
                                                                                          state.test_loader,
                                                                                          mode)
                explanations: List[Tensor] = [x for [x, _] in labeled_explanations]
                intra_class_variances_per_model[mode] = intra_class_variances(labeled_explanations)
                aggregated_variance_per_model[mode] = variance(explanations)
            state.intra_class_variances.append(intra_class_variances_per_model)
            state.aggregated_variances.append(aggregated_variance_per_model)

            state.accuracies.append(utils.compute_accuracy(state.explainers[model_nr].classifier, state.test_loader))
        state.visualization_loaders = get_visualization_loaders()

    accuracies = st.session_state.accuracies
    visualization_loaders = st.session_state.visualization_loaders

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)

    explanation_mode = st.sidebar.select_slider(label="Mode",
                                                options=modes)

    inputs, labels = iter(visualization_loaders[label]).__next__()

    column_1, column_2 = st.columns(2)

    for model_nr, column in enumerate([column_1, column_2]):  # compare two models
        with column:
            f"### Model {model_nr} "
            f"Trained on: `{state.explainers[model_nr].explanation_mode}`"
            f" Prediction: `{state.explainers[model_nr].predict(inputs)}`"
            f"accuracy: `{accuracies[model_nr]}`"
            st.write(f"Intra-Class Variance of Class `{label}` on {explanation_mode}:"
                     f" `{state.intra_class_variances[model_nr][explanation_mode][label]:.3f}`")
            mean_dist = mean(state.intra_class_variances[model_nr][explanation_mode])
            aggregated = state.aggregated_variances[model_nr][explanation_mode]
            f"Intra-Class Variance, averaged over classes `{mean_dist:.3f}`"
            f"Aggregated Variance: `{aggregated:.3f}`"
            f"Ratio `{mean_dist:.3f}/{aggregated:.3f} = {mean_dist/aggregated:.3f}`"

            f" Mode: `{explanation_mode}`"

            inputs = transform(inputs, "normalize")
            explanation_batch = state.explainers[model_nr].get_explanation_batch(inputs, labels, explanation_mode)

            explanation_batch = transform(explanation_batch, "unnormalize")
            for i in range(2):
                st.image(transforms.ToPILImage()(explanation_batch[i][0].squeeze_(0)), width=200, output_format='PNG')


def transform(images: Tensor, mode: str) -> Tensor:
    """Normalizes or unnormalizes images

    If the images are already normalized/unnormalized, leave them unchanged.

    :param images: images that shall be transformed.
    :param mode: "unnormalize" or "normalize" respectively if the images are supposed to be normalized or unnormalized.

    :returns: transformed images
    """

    # check if images are currently normalized or not
    normalized = not (images >= 0).all()

    mean_mnist: float = 0.1307
    std_dev_mnist: float = 0.3081

    if normalized and mode == "unnormalize":
        return images.mul(std_dev_mnist).add(mean_mnist).detach()
    if not normalized and mode == "normalize":
        return images.sub(mean_mnist).div(std_dev_mnist).detach()
    else:
        return images


@st.cache(allow_output_mutation=True)
def get_visualization_loaders():
    full_test_set = utils.FastMNIST('./data', train=False, download=True)
    # loads the data to the ./data folder
    # full_test_set.un_normalize()
    visualization_loaders = []
    for label in range(10):
        subset = Subset(full_test_set, torch.where(full_test_set.targets == label)[0][:4])
        visualization_loaders.append(DataLoader(subset, batch_size=2, num_workers=0))

    return visualization_loaders


@st.cache(allow_output_mutation=True)
def set_up_evaluation_experiments() -> Tuple[List[Explainer], DataLoader[Any], DataLoader[Any], str]:
    device: str
    cfg: SimpleArgumentParser
    cfg, device, *_ = utils.setup([], eval_mode=True)

    model_paths = ["fixed_testset_default_aso.pt", "pretrained_model.pt"]
    explainers: List[Explainer] = [get_empty_explainer(device=device, explanation_mode="input_x_gradient"),
                                   get_empty_explainer(device=device, explanation_mode="input")]

    for i in range(2):
        explainers[i].load_state(f"models/{model_paths[i]}")
        explainers[i].classifier.eval()

    # get the full 10000 MNIST test samples
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=1000,
                              batch_size=100,
                              test_batch_size=100)

    return explainers, loaders.test, loaders.visualization, device


def get_labeled_explanations(explainer: Explainer, test_loader: DataLoader, mode: str) -> List[Tuple[Tensor, int]]:
    """get all explanations together with the labels, and don't combine them into batches."""
    labeled_explanations = []
    for inputs, labels in test_loader:
        explanation_batch: List[Tensor] = list(explainer.get_explanation_batch(inputs, labels, mode))
        labeled_explanation_batch: List[Tuple[Tensor, int]] = list(zip(explanation_batch, list(labels)))
        labeled_explanations.extend(labeled_explanation_batch)
    return labeled_explanations


def intra_class_variances(labeled_points: List[Tuple[Tensor, int]]) -> List[float]:
    """sorts the points by their labels, and returns a list of the variances by label """
    intraclass_variances = []
    for label in range(10):
        label_subset = [point for [point, lab] in labeled_points if lab == label]
        intraclass_variances.append(variance(label_subset))
    return intraclass_variances


def variance(points: List[Tensor]) -> float:
    """computes the variance of a cluster of points"""
    points = torch.stack(points)
    mean_point = torch.mean(points, dim=0)
    differences_to_mean = points - mean_point
    # take the l_2 distance along the dimensions of the image
    l_2_distances = torch.norm(differences_to_mean, p=2, dim=[2, 3])
    return torch.mean(l_2_distances).item()


def get_empty_explainer(device, explanation_mode) -> Explainer:
    return Explainer(device=device, loaders=None, optimizer_type=None, logging=None, test_batch_to_visualize=None,
                     rtpt=None, model_path="", explanation_mode=explanation_mode)


if __name__ == '__main__':
    run_evaluation_experiments()
