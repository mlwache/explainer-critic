import os
from typing import Union, Iterable, Callable, Optional

import torch
import torch.optim as optim
from torch import Tensor
from torch.nn.modules import Module
import torch.nn as nn
from rtpt import RTPT
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from time import time

from tap import Tap


def get_device():
    if not torch.cuda.is_available():
        print(f"No GPU found, falling back to CPU.")
        return "cpu"
    else:
        return "cuda"


def make_rtpt(rtpt_enabled: bool, n_training_batches: int) -> Optional[RTPT]:
    if rtpt_enabled:
        return RTPT(name_initials='mwache',
                    experiment_name='explainer-critic',
                    max_iterations=n_training_batches)
    else:
        return None


def get_optimizer(learning_rate, momentum) -> Callable[[Union[Iterable[Tensor], Iterable[dict]]], SGD]:
    return lambda parameters: optim.SGD(parameters, lr=learning_rate, momentum=momentum)


class SimpleArgumentParser(Tap):
    # Training Details
    batch_size: int = 64
    learning_rate: float = 0.001
    momentum: float = 0.9

    # Dataset sizes
    n_training_batches: int = 50
    n_critic_batches: int = 50
    n_test_batches: int = 5

    render_enabled: bool = False
    rtpt_enabled: bool = False

    # number of samples follow accordingly
    n_training_samples: int = n_training_batches * batch_size
    n_critic_samples: int = n_critic_batches * batch_size
    n_test_samples: int = n_test_batches * batch_size

    # config values that are rarely changed
    PATH_TO_MODELS: str = './models/mnist_net.pth'
    TIMES_TO_PRINT_CRITIC: int = 10
    CLASSES: list = list(range(10))
    LOSS: Module = nn.CrossEntropyLoss()
    DEVICE: str = get_device()
    RTPT_OBJECT: RTPT = make_rtpt(rtpt_enabled, n_training_batches)
    MNIST_TOTAL_SAMPLES: int = 20000
    LOG_DIR: str = f"./runs/{int(time())}"
    WRITER: SummaryWriter = None
    OPTIMIZER: Callable[[Union[Iterable[Tensor], Iterable[dict]]], SGD] = None

    def process_args(self):
        n_total_samples = self.n_training_samples + self.n_test_samples + self.n_critic_samples
        assert n_total_samples <= self.MNIST_TOTAL_SAMPLES, f"{n_total_samples} in total are too much."
        self.WRITER: SummaryWriter = SimpleArgumentParser.make_tensorboard_writer(self.LOG_DIR)
        self.OPTIMIZER = get_optimizer(self.learning_rate, self.momentum)

    @staticmethod
    def make_tensorboard_writer(log_dir: str) -> SummaryWriter:
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir)
