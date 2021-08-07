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
    n_training_batches: int  # should be <= 20000/batch_size
    n_test_batches: int
    n_critic_batches: int
    n_epochs: int
    loss: Module
    render_enabled: bool
    rtpt_enabled: bool
    device: str

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
    def optimizer(self) -> object:
        return lambda parameters: optim.SGD(parameters, lr=self.learning_rate, momentum=self.momentum)


default_config = Config(batch_size=4,
                        learning_rate=0.001,
                        momentum=0.9,
                        path_to_models='./models/mnist_net.pth',
                        classes=list(range(10)),
                        n_training_batches=2,
                        n_test_batches=5,
                        n_critic_batches=2,
                        n_epochs=1,
                        loss=nn.CrossEntropyLoss(),
                        render_enabled=False,
                        rtpt_enabled=False,
                        device="cuda"
                        )

test_config = Config(batch_size=default_config.batch_size,
                     learning_rate=default_config.learning_rate,
                     momentum=default_config.momentum,
                     path_to_models=default_config.path_to_models,
                     classes=default_config.classes,
                     n_training_batches=4,
                     n_test_batches=2,
                     n_critic_batches=2,
                     n_epochs=1,
                     loss=default_config.loss,
                     render_enabled=False,
                     rtpt_enabled=False,
                     device=default_config.device
                     )
