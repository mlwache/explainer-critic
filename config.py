from dataclasses import dataclass, field
from typing import Union, Iterable, Callable

import torch
import torch.optim as optim
from torch import Tensor
from torch.nn.modules import Module
import torch.nn as nn
from rtpt import RTPT
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from time import time


@dataclass
class Config:
    name: str
    render_enabled: bool
    rtpt_enabled: bool

    # Training Details
    batch_size: int
    learning_rate: float
    momentum: float
    n_epochs: int

    # Dataset sizes
    n_training_batches: int
    n_test_batches: int
    n_critic_batches: int

    # config values that aren't set when calling
    PATH_TO_MODELS: str = field(init=False)
    CLASSES: list = field(init=False)
    LOSS: Module = field(init=False)
    DEVICE: str = field(init=False)
    RTPT_OBJECT: RTPT = field(init=False)
    MNIST_TOTAL_SAMPLES: int = field(init=False)
    WRITER: SummaryWriter = field(init=False)
    LOG_DIR: str = field(init=False)

    def __post_init__(self):
        self.PATH_TO_MODELS = './models/mnist_net.pth'
        self.LOG_DIR = f"./runs/{int(time())}"
        self.CLASSES = list(range(10))  # We are dealing with MNIST.
        self.LOSS = nn.CrossEntropyLoss()
        self.RTPT_OBJECT = RTPT(name_initials='mwache',
                                experiment_name='explainer-critic',
                                max_iterations=self.n_training_batches)
        self.MNIST_TOTAL_SAMPLES = 20000
        self.TIMES_TO_PRINT_CRITIC = 10
        n_total_samples = self.n_training_samples + self.n_training_samples + self.n_critic_samples
        assert n_total_samples <= self.MNIST_TOTAL_SAMPLES, f"MNIST only has {self.MNIST_TOTAL_SAMPLES} samples."
        self.set_device()

    @property
    def n_test_samples(self) -> int:
        return self.n_test_batches * self.batch_size

    @property
    def n_training_samples(self) -> int:
        return self.n_training_batches * self.batch_size

    @property
    def n_critic_samples(self) -> int:
        return self.n_critic_batches * self.batch_size

    @property
    def optimizer(self) -> Callable[[Union[Iterable[Tensor], Iterable[dict]]], SGD]:
        return lambda parameters: optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)

    def set_device(self):
        if not torch.cuda.is_available():
            print(f"Setting up {self.name}: No GPU found, falling back to CPU.")
            self.DEVICE = "cpu"
        else:
            self.DEVICE = "cuda"


default_config = Config(name="default_config",
                        batch_size=4,
                        learning_rate=0.001,
                        momentum=0.9,
                        n_training_batches=2,
                        n_test_batches=5,
                        n_critic_batches=2,
                        n_epochs=1,
                        render_enabled=False,
                        rtpt_enabled=False,
                        )

test_config = Config(name="test_config",
                     batch_size=default_config.batch_size,
                     learning_rate=default_config.learning_rate,
                     momentum=default_config.momentum,
                     n_training_batches=4,
                     n_test_batches=2,
                     n_critic_batches=2,
                     n_epochs=1,
                     render_enabled=False,
                     rtpt_enabled=False,
                     )

critic_test_config = Config(name="critic_test_config",
                            batch_size=80,
                            learning_rate=0.001,
                            momentum=0.9,
                            n_training_batches=2,
                            n_test_batches=5,
                            n_critic_batches=100,
                            n_epochs=1,
                            render_enabled=False,
                            rtpt_enabled=False,
                            )
