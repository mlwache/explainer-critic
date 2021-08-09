from dataclasses import dataclass, field
import torch.optim as optim
from torch.nn.modules import Module
import torch.nn as nn
from rtpt import RTPT

@dataclass()
class Config:
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
    path_to_models: str = field(init=False)
    classes: list = field(init=False)
    loss: Module = field(init=False)
    device: str = field(init=False)
    rtpt: RTPT = field(init=False)

    def __post_init__(self):
        self.path_to_models = './models/mnist_net.pth'
        self.classes = list(range(10))  # We are dealing with MNIST.
        self.loss = nn.CrossEntropyLoss()
        self.device = "cuda"
        self.rtpt = RTPT(name_initials='MW', experiment_name='Explainer-Critic', max_iterations=self.n_training_batches)
        n_total_samples = self.n_training_samples + self.n_training_samples + self.n_critic_samples
        assert n_total_samples <= 20000  # MNIST only has 20000 samples in total.

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
                        n_training_batches=2,
                        n_test_batches=5,
                        n_critic_batches=2,
                        n_epochs=1,
                        render_enabled=False,
                        rtpt_enabled=False,
                        )

test_config = Config(batch_size=default_config.batch_size,
                     learning_rate=default_config.learning_rate,
                     momentum=default_config.momentum,
                     n_training_batches=4,
                     n_test_batches=2,
                     n_critic_batches=2,
                     n_epochs=1,
                     render_enabled=False,
                     rtpt_enabled=False,
                     )
