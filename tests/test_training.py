from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import main
from config import SimpleArgumentParser
from experiments.experiments import train_critic_without_explanations, train_explainer_only_classification


@pytest.fixture
def args() -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    args.parse_args(args=[])
    return args


def test_main_load_data(args):
    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    critic_loader: DataLoader[Any]
    train_loader, test_loader, critic_loader = main.load_data(args)
    train_data_sample: Tensor
    for i, train_data_sample in enumerate(train_loader):
        assert i <= args.n_training_batches
        images, labels = train_data_sample
        assert len(images) == args.batch_size
        assert images[0].size() == torch.Size([1, 28, 28])
        assert torch.all(images[0].data.ge(-1*torch.ones_like(images)))
        # the upper bound is higher, as the images are normalized and there are less white/high-value pixels
        assert torch.all(images[0].data.le(5*torch.ones_like(images)))


def test_critic_makes_progress_without_explanations(args):
    initial_loss, end_of_training_loss = train_critic_without_explanations(args)
    assert abs(initial_loss - np.log(len(args.CLASSES))) < 0.1
    assert initial_loss - end_of_training_loss > 0.003


def test_explainer_makes_progress_with_only_classification(args):
    initial_loss, end_of_training_loss = train_explainer_only_classification(args)
    assert abs(initial_loss - np.log(len(args.CLASSES))) < 0.1
    assert initial_loss - end_of_training_loss > 0.003
