from tap import Tap


class SimpleArgumentParser(Tap):
    training_mode: str = "pretrain"

    # Training Details
    batch_size: int = 64
    learning_rate_start: float = 1.0
    learning_rate_step: float = 0.7
    pretrain_learning_rate: float = 1.0

    # Dataset sizes
    n_training_batches: int = 50
    n_critic_batches: int = 50
    n_test_batches: int = 5
    n_epochs: int = 10
    n_pretraining_epochs: int = 10

    render_enabled: bool = False
    rtpt_enabled: bool = False

    # config values that are rarely changed
    CLASSES: list = list(range(10))
    MNIST_TOTAL_SAMPLES: int = 20000

    MEAN_MNIST: float = 0.1307
    STD_DEV_MNIST: float = 0.3081

    def process_args(self):
        self.assert_not_too_many_samples()

    def assert_not_too_many_samples(self):
        n_total_training_samples = self.n_training_samples + self.n_critic_samples
        assert n_total_training_samples <= self.MNIST_TOTAL_SAMPLES, f"{n_total_training_samples} are too much."

    @property
    def n_training_samples(self) -> int:
        return self.n_training_batches * self.batch_size

    @property
    def n_critic_samples(self) -> int:
        return self.n_critic_batches * self.batch_size

    @property
    def n_test_samples(self) -> int:
        return self.n_test_batches * self.batch_size
