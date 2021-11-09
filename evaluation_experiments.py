import os
from typing import Any, Tuple

import streamlit as st
from torch.utils.data import DataLoader

import utils
from config import SimpleArgumentParser
from explainer import Explainer


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained.
    Expects that the arguments are the same as for the model that should be evaluated"""

    explainer, test_loader, device = set_up_evaluation_experiments()

    inputs, labels = iter(test_loader).__next__()

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

    st.selectbox(label="choose an explanation type", options=["input x grad", "integrated_grad"],
                 on_change=choose_explanation_type)


def choose_explanation_type():
    pass


def set_up_evaluation_experiments() -> Tuple[Explainer, DataLoader[Any], str]:
    # device: str
    # cfg: SimpleArgumentParser
    cfg, device, *_ = utils.setup([], eval_mode=True)

    explainer = Explainer(device, loaders=None, optimizer_type=None, logging=None, test_batch_to_visualize=None,
                          rtpt=None, model_path="")

    model_path = get_model_path(cfg)
    explainer.load_state(f"models/{model_path}")
    explainer.classifier.eval()

    # get the full 10000 MNIST test samples
    test_loader = utils.load_data(1, 1, 10000, batch_size=100).test

    return explainer, test_loader, device


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
