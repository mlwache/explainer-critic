from typing import Any, Tuple, List

import streamlit as st
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import utils
from config import SimpleArgumentParser
from explainer import Explainer


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained.
    Expects that the arguments are the same as for the model that should be evaluated"""

    explainers, test_loader, visualization_loader, device = set_up_evaluation_experiments()
    accuracies = compute_accuracies(explainers, test_loader)
    visualization_loaders = get_visualization_loaders()

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)

    explanation_mode = st.sidebar.select_slider(label="Explanation Mode",
                                                options=["gradient",
                                                         "input_x_gradient",
                                                         "integrated_gradient",
                                                         "input_x_integrated_gradient"])

    inputs, labels = iter(visualization_loaders[label]).__next__()

    inputs = transform(inputs, "unnormalize")
    for i in range(2):
        st.image(transforms.ToPILImage()(inputs[i][0].squeeze_(0)), width=100, output_format='PNG')

    column_1, column_2 = st.columns(2)
    column_1.write(f"### Model 1: Trained on {explainers[1].explanation_mode}")
    column_2.write("### Model 2: Trained on input")

    for model_nr, column in enumerate([column_1, column_2]):  # compare two models
        with column:

            f" Prediction: `{explainers[model_nr].predict(inputs)}`"
            # TODO
            # @st.cache  #don't compute this every time
            # ics = inter_class_similarity
            # st.markdown(f"Euclidean ICS of label {label}: {ics}")
            # st.markdown(f"Mean Euclidean ICS: {ics}")
            st.write(f"accuracy: {accuracies[model_nr]}")
            explainers[model_nr].explanation_mode = explanation_mode
            f" Explanation Mode: `{explanation_mode}`"

            inputs = transform(inputs, "normalize")
            explanations = explainers[model_nr].get_explanation_batch(inputs, labels)

            explanations = transform(explanations, "unnormalize")
            for i in range(2):
                st.image(transforms.ToPILImage()(explanations[i][0].squeeze_(0)), width=100, output_format='PNG')


@st.cache
def compute_accuracies(explainers, test_loader: DataLoader[Any]) -> List[float]:
    return [explainer.compute_accuracy(test_loader, n_batches=len(test_loader)) for explainer in explainers]

def transform(images: Tensor, mode: str) -> Tensor:
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
                              n_test_samples=10000,
                              batch_size=100,
                              test_batch_size=100)

    return explainers, loaders.test, loaders.visualization, device


def get_empty_explainer(device, explanation_mode) -> Explainer:
    return Explainer(device=device, loaders=None, optimizer_type=None, logging=None, test_batch_to_visualize=None,
                          rtpt=None, model_path="", explanation_mode=explanation_mode)



if __name__ == '__main__':
    run_evaluation_experiments()

# def get_model_path(cfg: SimpleArgumentParser):
#     # check if there is a model with the right name.
#     cfg_string = utils.config_string(cfg)
#     date_len: int = len(utils.date_time_string())
#     cfg_string = cfg_string[0:-date_len]  # ignore date at the end of config string
#     for model_name in os.listdir("models"):
#         if model_name.startswith(cfg_string):
#             return model_name
#     # otherwise fall back to pretrained model
#     pretrained_path = 'pretrained_model.pt'
#     if pretrained_path in os.listdir("models"):
#         return pretrained_path
#
#     raise ValueError(f"no model with name {cfg_string}_<date>.pt found, nor pretrained model.")
