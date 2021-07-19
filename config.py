from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int
    learning_rate: float
    momentum: float
    path_to_models: str
    classes: list
    n_training_samples: int
    n_test_samples: int
    n_epochs: int

    @property
    def n_test_batches(self) -> int:
        return self.n_test_samples // self.batch_size

    @property
    def n_train_batches(self) -> int:
        return self.n_training_samples // self.batch_size


default_config = Config(batch_size=4,
                        learning_rate=0.001,
                        momentum=0.9,
                        path_to_models='./models/mnist_net.pth',
                        classes=list(range(10)),
                        n_training_samples=2000,
                        n_test_samples=100,
                        n_epochs=1)
