from statistics import mean
from typing import Tuple, List, Dict

import streamlit as st
import torch
from streamlit import session_state as state
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import utils
from evaluation_experiments_console import set_up_evaluation_experiments, variance, intra_class_variances


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained."""
    modes = ["gradient", "input_x_gradient", "input"]
    if 'explainers' not in st.session_state:  # this part will run only once in the beginning.
        state.show_ics = False  # I'm not fully convinced that this is the right metric, so don't use it for now.
        state.n_models = 3
        state.explainers, state.test_loader, state.model_names = \
            set_up_evaluation_experiments(state.n_models, n_test_samples=1000)
        all_unnormalized_images = transform(utils.loader_to_tensors(state.test_loader)[0], "unnormalize")

        # all of the following are lists, because I compute them for multiple models.
        state.accuracies = []
        state.intra_class_variances = []
        state.aggregated_variances = []
        state.means = []

        for model_nr in range(state.n_models):
            means_for_this_model: Dict[str, float] = {}
            intra_class_variances_in_this_model: Dict[str, List[float]] = {}
            aggregated_variance_in_this_model: Dict[str, float] = {}
            for mode in modes:
                # first, get  all explanations/inputs of the test set
                explanations_tensor, labels_tensor = state.explainers[model_nr].get_labeled_explanations(
                    state.test_loader,
                    mode)
                means_for_this_model[mode] = torch.mean(torch.abs(explanations_tensor)).item()
                means_for_this_model["input"] = torch.mean(torch.abs(all_unnormalized_images)).item()
                if state.show_ics:
                    intra_class_variances_in_this_model[mode] = intra_class_variances(explanations_tensor,
                                                                                      labels_tensor)
                    aggregated_variance_in_this_model[mode] = variance(explanations_tensor)
            state.means.append(means_for_this_model)
            state.intra_class_variances.append(intra_class_variances_in_this_model)
            state.aggregated_variances.append(aggregated_variance_in_this_model)

            state.accuracies.append(utils.compute_accuracy(state.explainers[model_nr].classifier, state.test_loader))
        state.visualization_loaders = get_visualization_loaders()

    # assert len(state.aggregated_variances) == state.n_models
    # assert len(state.aggregated_variances[0]) == len(modes)

    accuracies = st.session_state.accuracies
    visualization_loaders = st.session_state.visualization_loaders

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)
    n_img = st.sidebar.slider(label="Number of Images to show", min_value=1, max_value=10, value=2, step=1)

    explanation_mode = st.sidebar.select_slider(label="Mode", options=modes)

    input_types = ["black", "white", "gray", "noise"]
    input_types.extend(list(map(str, range(10))))
    input_type = st.sidebar.select_slider(label="Input Type", options=input_types)

    inputs = get_inputs(input_type, visualization_loaders, n_inputs=n_img)
    labels = torch.tensor([label] * n_img)

    columns = st.columns(state.n_models)

    for model_nr in range(state.n_models):  # compare two models
        with columns[model_nr]:
            f"### {state.model_names[model_nr][0:-3]} "
            f"Trained on: `{state.explainers[model_nr].explanation_mode}`"
            f" Prediction: `{state.explainers[model_nr].predict(inputs)}`"
            f"accuracy: `{accuracies[model_nr]}`"
            if state.show_ics:
                st.write(f"Intra-Class Mean Distance to Mean of Class `{label}` on {explanation_mode}:"
                         f" `{state.intra_class_variances[model_nr][explanation_mode][label]:.3f}`")
                mean_intra_class_variance = mean(state.intra_class_variances[model_nr][explanation_mode])

                aggregated = state.aggregated_variances[model_nr][explanation_mode]
                f"Intra-Class Mean Distance to Mean, averaged over classes `{mean_intra_class_variance:.3f}`"
                f"Aggregated Mean Distance to Mean: `{aggregated:.3f}`"
                f"Ratio `{aggregated:.3f}/{mean_intra_class_variance:.3f} = " \
                    f"{aggregated / mean_intra_class_variance:.3f}`"

            f"Mode: `{explanation_mode}`"
            f"Mean of `{explanation_mode}`: {state.means[model_nr][explanation_mode]:.5f}"
            assert abs(state.means[model_nr]["input"] - 0.1307) < 0.01

            inputs = transform(inputs, "normalize")
            state.explainers[model_nr].classifier.eval()
            explanation_batch = state.explainers[model_nr].get_explanation_batch(inputs, labels, explanation_mode)

            explanation_batch = transform(explanation_batch, "unnormalize")
            for i in range(n_img):
                st_show_tensor(explanation_batch[i][0].squeeze_(0))


def get_inputs(input_type: str, loaders: List[DataLoader], n_inputs: int) -> Tensor:
    if input_type.isdigit():
        inputs, _ = resize_batch(loader=loaders[int(input_type)], new_batch_size=n_inputs)
    else:
        raise NotImplementedError
    return inputs


def st_show_tensor(image: Tensor):
    st.image(transforms.ToPILImage()(image), width=200, output_format='PNG')


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


if __name__ == '__main__':
    run_evaluation_experiments()
