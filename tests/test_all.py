from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import main
import utils
from config import SimpleArgumentParser
from experiments import train_critic_without_explanations, train_explainer_only_classification, run_experiments


@pytest.fixture
def args() -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    args.parse_args(args=[])
    return args


def test_experiments_dont_crash():
    for training_mode in ["pretrain", "combined", "only_critic", "only_classification", "in_turns"]:
        run_experiments(['--batch_size=4', '--n_training_batches=2', '--n_critic_batches=2',
                         '--n_epochs=3', f'--training_mode={training_mode}'])


def test_main_load_data(args):
    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    critic_loader: DataLoader[Any]
    train_loader, test_loader, critic_loader = utils.load_data(args)
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
    n_classes = 10
    initial_loss, end_of_training_loss = train_critic_without_explanations(args, device=utils.get_device())
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.02


def test_explainer_makes_progress_with_only_classification(args):
    n_classes = 10
    initial_loss, end_of_training_loss = train_explainer_only_classification(args, device=utils.get_device())
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.01
