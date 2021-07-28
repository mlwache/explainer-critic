from dataclasses import dataclass
import torch.optim as optim
from torch.nn.modules import Module
import torch.nn as nn


@dataclass
class Config:
    batch_size: int
    learning_rate: float
    momentum: float
    path_to_models: str
    classes: list
    n_training_samples: int  # should be <= 20000
    n_test_samples: int
    n_critic_samples: int
    n_epochs: int
    loss: Module

    @property
    def n_test_batches(self) -> int:
        return self.n_test_samples // self.batch_size

    @property
    def n_train_batches(self) -> int:
        return self.n_training_samples // self.batch_size

    @property
    def n_critic_batches(self) -> int:
        return self.n_critic_samples // self.batch_size

    @property
    def optimizer(self) -> object:
        return lambda parameters: optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)


default_config = Config(batch_size=8,
                        learning_rate=0.001,
                        momentum=0.9,
                        path_to_models='./models/mnist_net.pth',
                        classes=list(range(10)),
                        n_training_samples=40,
                        n_test_samples=20,
                        n_critic_samples=8,
                        n_epochs=1,
                        loss=nn.CrossEntropyLoss()
                        )

test_config = Config(batch_size=default_config.batch_size,
                     learning_rate=default_config.learning_rate,
                     momentum=default_config.momentum,
                     path_to_models=default_config.path_to_models,
                     classes=default_config.classes,
                     n_training_samples=16,
                     n_test_samples=8,
                     n_critic_samples=8,
                     n_epochs=1,
                     loss=default_config.loss
                     )
