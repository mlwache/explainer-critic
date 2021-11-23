from typing import Any, Tuple

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

    explainer, test_loader, visualization_loader, device = set_up_evaluation_experiments()
    visualization_loaders = get_visualization_loaders()

    label = st.sidebar.slider(label="Label", min_value=0, max_value=9, value=5, step=1)
    inputs, labels = iter(visualization_loaders[label]).__next__()

    inputs = transform(inputs, "unnormalize")
    for i in range(2):
        st.image(transforms.ToPILImage()(inputs[i][0].squeeze_(0)), width=100, output_format='PNG')

    st.markdown(f" Prediction: `{explainer.predict(inputs)}`")
    st.markdown(f"Ground truth:`{labels}`")

    explanation_mode = st.sidebar.select_slider(label="Explanation Mode",
                                                options=["gradient",
                                                         "input_x_gradient",
                                                         "integrated_gradient",
                                                         "input_x_integrated_gradient"])
    explainer.explanation_mode = explanation_mode
    st.markdown(f" Explanation Mode: `{explanation_mode}`")

    inputs = transform(inputs, "normalize")
    explanations = explainer.get_explanation_batch(inputs, labels)

    explanations = transform(explanations, "unnormalize")
    for i in range(2):
        st.image(transforms.ToPILImage()(explanations[i][0].squeeze_(0)), width=100, output_format='PNG')


def transform(images: Tensor, mode: str) -> Tensor:
    # check if images are currently normalized or not
    normalized = not (images >= 0).all()

    mean_mnist: float = 0.1307
    std_dev_mnist: float = 0.3081

    if normalized and mode == "unnormalize":
        return images.mul_(std_dev_mnist).add_(mean_mnist)
    if not normalized and mode == "normalize":
        return images.sub_(mean_mnist).div_(std_dev_mnist)
    else:
        return images


def get_visualization_loaders():
    full_test_set = utils.FastMNIST('./data', train=False, download=True)
    # loads the data to the ./data folder
    # full_test_set.un_normalize()
    visualization_loaders = []
    for label in range(10):
        subset = Subset(full_test_set, torch.where(full_test_set.targets == label)[0][:4])
        visualization_loaders.append(DataLoader(subset, batch_size=2, num_workers=0))

    return visualization_loaders


def set_up_evaluation_experiments() -> Tuple[Explainer, DataLoader[Any], DataLoader[Any], str]:
    device: str
    cfg: SimpleArgumentParser
    cfg, device, *_ = utils.setup([], eval_mode=True)

    explainer = Explainer(device, loaders=None, optimizer_type=None, logging=None, test_batch_to_visualize=None,
                          rtpt=None, model_path="", explanation_mode=cfg.explanation_mode)

    model_path = "fixed_testset_default_aso.pt"  # get_model_path(cfg)
    explainer.load_state(f"models/{model_path}")
    explainer.classifier.eval()

    # get the full 10000 MNIST test samples
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=10000,
                              batch_size=100,
                              test_batch_size=100)

    return explainer, loaders.test, loaders.visualization, device,


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
