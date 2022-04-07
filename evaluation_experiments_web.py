from statistics import mean
from typing import Any, Tuple, List, Dict, Optional

import streamlit as st
import torch
from streamlit import session_state as state
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import utils
from explainer import Explainer


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained."""
    modes = ["gradient", "input_x_gradient", "input"]
    if 'explainers' not in st.session_state:  # this part will run only once in the beginning.
        state.n_models = 3
        state.explainers, state.test_loader, state.model_names = \
            set_up_evaluation_experiments(state.n_models)

        # all of the following are lists, because I compute them for multiple models.
        state.accuracies = []
        state.intra_class_variances = []
        state.aggregated_variances = []

        for model_nr in range(state.n_models):
            intra_class_variances_in_this_model: Dict[str, List[float]] = {}
            aggregated_variance_in_this_model: Dict[str, float] = {}
            for mode in modes:
                # first, get  all explanations/inputs of the test set
                labeled_explanations: List[Tuple[Tensor, int]] = get_labeled_explanations(state.explainers[model_nr],
                                                                                          state.test_loader,
                                                                                          mode)
                explanations: List[Tensor] = [x for [x, _] in labeled_explanations]
                intra_class_variances_in_this_model[mode] = intra_class_variances(labeled_explanations)
                aggregated_variance_in_this_model[mode] = variance(explanations)
            state.intra_class_variances.append(intra_class_variances_in_this_model)
            state.aggregated_variances.append(aggregated_variance_in_this_model)

            state.accuracies.append(utils.compute_accuracy(state.explainers[model_nr].classifier, state.test_loader))
        state.visualization_loaders = get_visualization_loaders()

    assert len(state.aggregated_variances) == state.n_models
    assert len(state.aggregated_variances[0]) == len(modes)

    accuracies = st.session_state.accuracies
    visualization_loaders = st.session_state.visualization_loaders

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)
    n_img = st.sidebar.slider(label="Number of Images to show", min_value=1, max_value=10, value=2, step=1)

    explanation_mode = st.sidebar.select_slider(label="Mode",
                                                options=modes)

    inputs, labels = resize_batch(loader=visualization_loaders[label], new_batch_size=n_img)
    columns = st.columns(state.n_models)

    for model_nr in range(state.n_models):  # compare two models
        with columns[model_nr]:
            f"### {state.model_names[model_nr][0:-3]} "
            f"Trained on: `{state.explainers[model_nr].explanation_mode}`"
            f" Prediction: `{state.explainers[model_nr].predict(inputs)}`"
            f"accuracy: `{accuracies[model_nr]}`"
            st.write(f"Intra-Class Variance of Class `{label}` on {explanation_mode}:"
                     f" `{state.intra_class_variances[model_nr][explanation_mode][label]:.3f}`")
            mean_intra_class_variance = mean(state.intra_class_variances[model_nr][explanation_mode])

            aggregated = state.aggregated_variances[model_nr][explanation_mode]
            f"Intra-Class Variance, averaged over classes `{mean_intra_class_variance:.3f}`"
            f"Aggregated Variance: `{aggregated:.3f}`"
            f"Ratio `{aggregated:.3f}/{mean_intra_class_variance:.3f} = {aggregated / mean_intra_class_variance:.3f}`"

            f" Mode: `{explanation_mode}`"

            inputs = transform(inputs, "normalize")
            state.explainers[model_nr].classifier.eval()
            explanation_batch = state.explainers[model_nr].get_explanation_batch(inputs, labels, explanation_mode)

            explanation_batch = transform(explanation_batch, "unnormalize")
            for i in range(n_img):
                st.image(transforms.ToPILImage()(explanation_batch[i][0].squeeze_(0)), width=200, output_format='PNG')


def resize_batch(loader: DataLoader, new_batch_size: int) -> Tuple[Tensor, Tensor]:
    new_input_batch_list: List[Tensor] = []
    new_label_batch_list: List[Tensor] = []
    old_batch_size: int = loader.batch_size
    for i, (input_batch, label_batch) in enumerate(loader):
        if len(new_input_batch_list) + old_batch_size > new_batch_size:
            # if adding another batch would lead to a higher batch size than the target bach size, only add a part.
            new_label_batch_list.extend(list(label_batch)[0:new_batch_size % old_batch_size])
            new_input_batch_list.extend(list(input_batch)[0:new_batch_size % old_batch_size])
            break
        new_input_batch_list.extend(list(input_batch))
        new_label_batch_list.extend(list(label_batch))
    new_input_batch: Tensor = torch.stack(new_input_batch_list)
    new_label_batch: Tensor = torch.stack(new_label_batch_list)

    assert len(new_label_batch) == new_batch_size, f"len={len(new_label_batch)}"

    return new_input_batch, new_label_batch


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


def get_visualization_loaders() -> List[DataLoader]:
    """Returns a list of data loaders for visualization - one for each label"""
    full_test_set = utils.FastMNIST('./data', train=False, download=True)
    # loads the data to the ./data folder
    visualization_loaders = []
    for label in range(10):
        # for visualization, we only need to load 10 images for each label, more would be confusing
        subset = Subset(full_test_set, torch.where(full_test_set.targets == label)[0][:10])
        visualization_loaders.append(DataLoader(subset, batch_size=2))

    return visualization_loaders


def set_up_evaluation_experiments(n_models: int,
                                  run_name: Optional[str] = None,
                                  loaders=None,
                                  used_for_training=False,
                                  ) -> Tuple[List[Explainer],
                                             DataLoader[Any],
                                             List[str]]:
    args = utils.parse_args(overriding_args=[f'--run_name={run_name}'])
    utils.setup(args, eval_mode=not used_for_training)

    model_paths = ["trained_explainer.pt",
                   "pre-trained.pt"]

    explanation_modes = ["input_x_gradient",
                         "input",
                         "nothing"][0:n_models]
    explainers: List[Explainer] = get_list_of_empty_explainers(explanation_modes=explanation_modes,
                                                               loaders=loaders)
    for i in range(n_models):
        if i < len(model_paths):
            explainers[i].load_state(f"models/{model_paths[i]}")
        else:
            print("Not enough models paths specified, using un-trained model instead.")
            model_paths.append("un-trained.pt")
        explainers[i].classifier.eval()

    # get the full 10000 MNIST test samples
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=1000,
                              batch_size=100,
                              test_batch_size=100)

    return explainers, loaders.test, model_paths


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
    intraclass_variances: List[float] = []
    for label in range(10):
        label_subset = [point for [point, lab] in labeled_points if lab == label]
        intraclass_variances.append(variance(label_subset))
    return intraclass_variances


def variance(points: List[Tensor]) -> float:
    """computes the variance of a cluster of points"""
    points = torch.stack(points)
    mean_point = torch.mean(points, dim=0)
    differences_to_mean = points - mean_point
    differences_as_vector = torch.flatten(differences_to_mean, start_dim=1, end_dim=3)
    differences_square = differences_as_vector * differences_as_vector
    # take the l_2 distance along the dimensions of the image
    # l_2_distances = torch.norm(differences_to_mean, p=2, dim=[2, 3])
    l_2_distances = torch.sqrt(differences_square.sum(dim=1))
    return torch.mean(l_2_distances).item()


def get_list_of_empty_explainers(explanation_modes, loaders) -> List[Explainer]:
    return [Explainer(loaders=loaders,
                      optimizer_type=None,
                      test_batch_to_visualize=None,
                      model_path="",
                      explanation_mode=explanation_mode) for explanation_mode in explanation_modes]


if __name__ == '__main__':
    run_evaluation_experiments()
