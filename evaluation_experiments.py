from statistics import mean
from typing import Any, Tuple, List

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

    if 'explainers' not in st.session_state:
        state.explainers, state.test_loader, state.visualization_loader, state.device = set_up_evaluation_experiments()

        # compute and save the input mean distances
        data_as_list: List[Tuple[Tensor, int]] = data_to_list(state.test_loader)
        input_list: List[Tensor] = [x for [x, _] in data_as_list]
        state.input_intra_class_mean_distances = intra_class_mean_square_distances(data_as_list)
        state.input_aggregated_mean_distance = mean_square_distance(input_list)

        # all of the following are lists, because I compute them for multiple models.
        state.accuracies = []
        state.intra_class_mean_distances = []
        state.aggregated_mean_distances = []

        for model_nr in range(2):
            # first, get  all explanations of the test set
            labeled_explanations: List[Tuple[Tensor, int]] = get_labeled_explanations(state.explainers[model_nr],
                                                                                      state.test_loader)
            explanations: List[Tensor] = [x for [x, _] in labeled_explanations]

            state.intra_class_mean_distances.append(intra_class_mean_square_distances(labeled_explanations))
            state.aggregated_mean_distances.append(mean_square_distance(explanations))

            state.accuracies.append(state.explainers[model_nr].compute_accuracy(state.test_loader))
        state.visualization_loaders = get_visualization_loaders()

    accuracies = st.session_state.accuracies
    visualization_loaders = st.session_state.visualization_loaders

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)

    explanation_mode = st.sidebar.select_slider(label="Explanation Mode",
                                                options=["gradient",
                                                         "input_x_gradient",
                                                         "integrated_gradient",
                                                         "input_x_integrated_gradient"])

    inputs, labels = iter(visualization_loaders[label]).__next__()

    inputs = transform(inputs, "unnormalize")
    for i in range(2):
        st.image(transforms.ToPILImage()(inputs[i][0].squeeze_(0)), width=200, output_format='PNG')

    st.write(f"Intra-Class Mean Square Distance of Class `{label}` on input:"
             f" `{state.input_intra_class_mean_distances[label]:.3f}`")
    st.write(f"Intra-Class MSD, averaged over classes on input: `{mean(state.input_intra_class_mean_distances):.3f}`")
    st.write(f"Aggregated Mean Square Distance on input: `{state.input_aggregated_mean_distance:.3f}`")


    column_1, column_2 = st.columns(2)
    column_1.write(f"### Model 1: Trained on {state.explainers[1].explanation_mode}")
    column_2.write("### Model 2: Trained on input")

    for model_nr, column in enumerate([column_1, column_2]):  # compare two models
        with column:
            st.write(f"Intra-Class Mean Square Distance of Class `{label}` on {state.explainers[model_nr].explanation_mode}:"
                     f" `{state.intra_class_mean_distances[model_nr][label]:.3f}`")
            st.write(f"Intra-Class MSD, averaged over classes `{mean(state.intra_class_mean_distances[model_nr]):.3f}`")
            st.write(f"Aggregated Mean Square Distance: `{state.aggregated_mean_distances[model_nr]:.3f}`")
            st.write(f"accuracy: `{accuracies[model_nr]}`")

            f" Prediction: `{state.explainers[model_nr].predict(inputs)}`"
            state.explainers[model_nr].explanation_mode = explanation_mode
            # TODO instead of changing this, use it as an input.
            f" Explanation Mode: `{explanation_mode}`"

            inputs = transform(inputs, "normalize")
            explanation_batch = state.explainers[model_nr].get_explanation_batch(inputs, labels)

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
    explainers: List[Explainer] = []
    for i in range(2):
        explainers.append(get_empty_explainer(device=device, explanation_mode=cfg.explanation_mode))
        explainers[i].load_state(f"models/{model_paths[i]}")
        explainers[i].classifier.eval()

    # get the full 10000 MNIST test samples
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=1000,
                              batch_size=100,
                              test_batch_size=100)

    return explainers, loaders.test, loaders.visualization, device


def get_labeled_explanations(explainer: Explainer, test_loader: DataLoader) -> List[Tuple[Tensor, int]]:
    """get all explanations together with the labels, and don't combine them into batches."""
    labeled_explanations = []
    for inputs, labels in test_loader:
        explanation_batch: List[Tensor] = list(explainer.get_explanation_batch(inputs, labels))
        labeled_explanation_batch: List[Tuple[Tensor, int]] = list(zip(explanation_batch, list(labels)))
        labeled_explanations.extend(labeled_explanation_batch)
    return labeled_explanations


def data_to_list(loader: DataLoader) -> List[Tuple[Tensor, int]]:
    data = []
    for inputs, labels in loader:
        labeled_batch: List[Tuple[Tensor, int]] = list(zip(list(inputs), list(labels)))
        data.extend(labeled_batch)
    return data


def intra_class_mean_square_distances(labeled_points: List[Tuple[Tensor, int]]) -> List[float]:
    """sorts the points by their labels, and returns a list of the mean square distances by label """
    intraclass_mean_square_distances = []
    for label in range(10):
        label_subset = [point for [point, lab] in labeled_points if lab == label]
        intraclass_mean_square_distances.append(mean_square_distance(label_subset))
    return intraclass_mean_square_distances


def mean_square_distance(points: List[Tensor]) -> float:
    """computes the mean square distance to the mean of a cluster of points"""
    points = torch.stack(points)
    # TODO check dims
    mean_point = torch.mean(points, dim=0)
    # TODO show mean point
    differences_to_mean = points - mean_point
    # take the l_2 distance along the dimensions of the image
    l_2_distances = torch.norm(differences_to_mean, p=2, dim=[2, 3])
    return torch.mean(l_2_distances).item()


def get_empty_explainer(device, explanation_mode) -> Explainer:
    return Explainer(device=device, loaders=None, optimizer_type=None, logging=None, test_batch_to_visualize=None,
                     rtpt=None, model_path="", explanation_mode=explanation_mode)


if __name__ == '__main__':
    run_evaluation_experiments()
