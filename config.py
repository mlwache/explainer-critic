from tap import Tap


def _colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[0m".format(r, g, b, text)


class SimpleArgumentParser(Tap):
    training_mode: str = "pretrain"
    logging_disabled: bool = False
    constant_lr: bool = False

    # Training Details
    batch_size: int = 128
    learning_rate_start: float = 0.2
    learning_rate_step: float = 0.7
    learning_rate_critic: float = 0.2
    pretrain_learning_rate: float = 1.0

    # Dataset sizes
    n_training_batches: int = 400
    n_critic_batches: int = 68
    # these are the default values as the MNIST training set has 60000>468*128 samples.
    n_test_batches: int = 5
    n_epochs: int = 40
    n_pretraining_epochs: int = 10
    # 10 episodes are enough to converge

    log_interval: int = 1
    # in case some day explainer values seem too much (e.g. if Tensorboard is overburdened and slow).
    # Setting this to a different value than 1 will lead to the critic plot having somewhat confusing holes.
    log_interval_critic: int = 5
    log_interval_pretraining: int = log_interval
    # Setting this to a different value than log_interval will lead to pre- and combined training having
    # different logging intervals.
    log_interval_accuracy: int = 20
    # setting this to a lower value will reduce performance significantly.

    render_enabled: bool = False
    rtpt_enabled: bool = False

    # config values that are rarely changed
    CLASSES: list = list(range(10))

    MEAN_MNIST: float = 0.1307
    STD_DEV_MNIST: float = 0.3081

    def process_args(self):
        low_number_of_iterations = 50
        n_iterations = self.n_iterations
        if not self.logging_disabled and n_iterations < low_number_of_iterations:
            # if we have so few iterations then it's probably a debug run.
            print(_colored(200, 150, 0, f"Logging everything, as there are only {n_iterations} iterations"))
            self.log_interval = 1
            self.log_interval_critic = 1
            self.log_interval_pretraining = 1
            self.log_interval_accuracy = 1
        if self.batch_size <= 16:
            print(_colored(200, 150, 0, f"lowered pretraining learning rate as batch size {self.batch_size} is small"))
            self.pretrain_learning_rate = 0.2

    @property
    def combined_iterations(self) -> int:
        return self.n_epochs * self.n_training_batches * self.n_critic_batches

    @property
    def pretraining_iterations(self) -> int:
        return self.n_pretraining_epochs * self.n_training_batches

    @property
    def n_iterations(self) -> int:
        if self.training_mode == 'combined':
            return self.combined_iterations
        elif self.training_mode == 'pretrain':
            return self.pretraining_iterations + self.combined_iterations
        elif self.training_mode == 'only_critic':
            return self.n_critic_batches  # critic only trains one episode
        elif self.training_mode == 'only_classification':
            return self.pretraining_iterations
        elif self.training_mode == 'in_turns':
            raise NotImplementedError
        elif self.training_mode == 'one_critic_pass':
            return self.pretraining_iterations + self.n_critic_batches
        else:
            raise ValueError(f"invalid training mode: {self.training_mode}")

    @property
    def n_training_samples(self) -> int:
        return self.n_training_batches * self.batch_size

    @property
    def n_critic_samples(self) -> int:
        return self.n_critic_batches * self.batch_size

    @property
    def n_test_samples(self) -> int:
        return self.n_test_batches * self.batch_size
