import os
from typing import Any, Tuple

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

import utils
from config import SimpleArgumentParser
from explainer import Explainer


def run_evaluation_experiments():
    """Runs some experiments on the models that are already trained.
    Expects that the arguments are the same as for the model that should be evaluated"""

    explainer, test_loader, device = set_up_evaluation_experiments(batch_size=100)

    inputs, labels = iter(test_loader).__next__()
    inputs, labels = inputs.to(device), labels.to(device)

    print("prediction: ", explainer.predict(inputs))
    print("ground truth: ", labels)


def set_up_evaluation_experiments(batch_size: int) -> Tuple[Explainer, DataLoader[Any], str]:
    cfg, device, *_ = utils.setup([], eval_mode=True)

    explainer = Explainer(device, loaders=None, logging=None, test_batch_to_visualize=None, rtpt=None, model_path="")

    model_path = get_model_path(cfg)
    explainer.load_state(f"models/{model_path}")
    explainer.classifier.eval()

    # get the full 10000 MNIST test samples
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(),
         torchvision.transforms.Normalize((cfg.MEAN_MNIST,), (cfg.STD_DEV_MNIST,))
         ])
    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    test_loader = DataLoader(test_set, batch_size)

    return explainer, test_loader, device


def get_model_path(cfg: SimpleArgumentParser):
    # check if there is a model with the right name.
    cfg_string = utils.config_string(cfg)
    date_len: int = len(utils.date_time_string())
    cfg_string = cfg_string[0:-date_len]  # ignore date at the end of config string
    for model_name in os.listdir("models"):
        if model_name.startswith(cfg_string):
            return model_name
    raise ValueError(f"no model with name {cfg_string}_<date>.pt found.")


if __name__ == '__main__':
    run_evaluation_experiments()
