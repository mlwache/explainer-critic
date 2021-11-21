import os
from typing import Any, Tuple

import streamlit as st
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
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

    explanation_type = st.sidebar.select_slider(label="Explanation Type",
                                                options=["input x grad", "integrated_grad", "grad"])
    st.markdown(f" Explanation Type: `{explanation_type}`")

    st.image(transforms.ToPILImage()(inputs[0][0].squeeze_(0)), width=100, output_format='PNG')

    progress_bar = st.sidebar.progress(0)

    for i in range(10):
        progress_bar.progress(i)

    st.markdown(f"""
    ### Prediction: 
    ```
    {explainer.predict(inputs)}
    ```""")
    st.markdown(f"""
    ### ground truth: 
    ```
    {labels} 
    ```""")


def get_visualization_loaders():
    full_test_set = utils.FastMNIST('./data', train=False, download=True)
    # loads the data to the ./data folder
    full_test_set.un_normalize()
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

    model_path = get_model_path(cfg)
    explainer.load_state(f"models/{model_path}")
    explainer.classifier.eval()

    # get the full 10000 MNIST test samples
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=10000,
                              batch_size=100,
                              test_batch_size=100)

    return explainer, loaders.test, loaders.visualization, device,


def get_model_path(cfg: SimpleArgumentParser):
    # check if there is a model with the right name.
    cfg_string = utils.config_string(cfg)
    date_len: int = len(utils.date_time_string())
    cfg_string = cfg_string[0:-date_len]  # ignore date at the end of config string
    for model_name in os.listdir("models"):
        if model_name.startswith(cfg_string):
            return model_name
    # otherwise fall back to pretrained model
    pretrained_path = 'pretrained_model.pt'
    if pretrained_path in os.listdir("models"):
        return pretrained_path

    raise ValueError(f"no model with name {cfg_string}_<date>.pt found, nor pretrained model.")


if __name__ == '__main__':
    run_evaluation_experiments()
