import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Tuple, Any, Optional, List

import numpy as np
import torch.cuda
import torch.multiprocessing
from rtpt import RTPT
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

from config import SimpleArgumentParser
from visualization import ImageHandler


@dataclass
class Loaders:
    train: DataLoader[Any]
    critic: Optional[DataLoader[Any]]
    test: Optional[DataLoader[Any]]
    visualization: Optional[DataLoader[Any]]


@dataclass
class Logging:
    """Combines the variables that are only used for logging"""
    writer: SummaryWriter
    run_name: str
    log_interval: int
    log_interval_accuracy: int
    n_test_batches: int
    critic_log_interval: int


def load_data_from_args(args: SimpleArgumentParser) -> Loaders:
    return load_data(n_training_samples=args.n_training_samples,
                     n_critic_samples=args.n_critic_samples,
                     n_test_samples=args.n_test_samples,
                     batch_size=args.batch_size,
                     test_batch_size=args.test_batch_size)


# noinspection PyShadowingNames
def load_data(n_training_samples: int,
              n_critic_samples: int,
              n_test_samples: int,
              batch_size: int,
              test_batch_size: int) -> Loaders:
    training_and_critic_set = FastMNIST('./data', train=True, download=True)
    full_test_set = FastMNIST('./data', train=False, download=True)
    # loads the data to the ./data folder

    # check that we have enough samples:
    n_total_training_samples = n_training_samples + n_critic_samples
    n_spare_samples = len(training_and_critic_set) - n_total_training_samples
    assert n_spare_samples >= 0, f"{n_total_training_samples} samples are too much. " \
                                 f"Please reduce the number of training or critic batches, or the batch size."

    # split training set into one training set for the classification, and one for the critic
    train_split = [n_training_samples, n_critic_samples, n_spare_samples]
    training_set, critic_set, _ = random_split(training_and_critic_set, train_split)

    # get a randomly split set for testing, and an ordered subset for the visualization
    test_split = [n_test_samples, len(full_test_set) - n_test_samples]

    test_set: Subset
    test_set, _ = random_split(full_test_set, test_split)
    # for the visualization get 50 samples of the dataset, 5 for each label
    visualization_sets = []
    for label in range(10):
        visualization_sets.append(Subset(full_test_set, torch.where(full_test_set.targets == label)[0][:4]))
    visualization_set = ConcatDataset(visualization_sets)
    n_vis_samples = visualization_set.cumulative_sizes[-1]

    loaders = Loaders(
        train=DataLoader(training_set, batch_size=batch_size, num_workers=0, shuffle=True),
        # for the critic set, I do the shuffling explicitly during training
        # in order to match samples to their respective explanations.
        critic=DataLoader(critic_set, batch_size=batch_size, num_workers=0),
        test=DataLoader(test_set, batch_size=test_batch_size, num_workers=0),
        visualization=DataLoader(visualization_set, batch_size=n_vis_samples, num_workers=0))

    return loaders


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def get_device():
    if not torch.cuda.is_available():
        print(colored(200, 150, 0, f"No GPU found, falling back to CPU."))
        return "cpu"
    else:
        return "cuda"


def set_sharing_strategy():
    # The following prevents there being too many open files at dl1.
    torch.multiprocessing.set_sharing_strategy('file_system')


def write_config_to_log(args: SimpleArgumentParser, log_dir):
    # Write config to log file
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json_dump: str = json.dumps(args.__dict__, default=lambda o: '<not serializable>')
        f.write(json_dump)
    # TODO: use typed_argparse's save.


def date_time_string():
    date_time = str(datetime.now())[0:-7]
    return date_time.replace(" ", "_")


def config_string(cfg: SimpleArgumentParser) -> str:
    lr_mode = "_sched" if cfg.lr_scheduling else ""

    # just for somewhat nicer formatting:
    run_name = cfg.run_name + "_" if cfg.run_name else ""

    return f'{run_name}' \
           f'{cfg.explanation_mode}' \
           f'_{cfg.training_mode}_ex{cfg.n_training_batches}_cr{cfg.n_critic_batches}' \
           f'_lr{cfg.learning_rate}{lr_mode}' \
           f'_bs{cfg.batch_size}_ep{cfg.n_epochs}_p-ep{cfg.n_pretraining_epochs}' \
           f'_gm{cfg.learning_rate_step}_ts{cfg.n_test_batches}' \
           f'_lr-c{cfg.learning_rate_critic}' \
           f'_lambda{cfg.explanation_loss_weight}' \
           f'_{date_time_string()}'


def get_one_batch_of_images(device: str, loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    return images, labels


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FastMNIST(MNIST):
    # code snippet from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist

    MEAN_MNIST: float = 0.1307
    STD_DEV_MNIST: float = 0.3081

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(self.MEAN_MNIST).div_(self.STD_DEV_MNIST)

        # Put both data and targets on GPU in advance
        device = get_device()
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def un_normalize(self):
        self.data = self.data.mul_(self.STD_DEV_MNIST).add_(self.MEAN_MNIST)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def setup(overriding_args: Optional[List], eval_mode: bool = False) -> Tuple[SimpleArgumentParser, str, Logging,
                                                                             Optional[RTPT]]:
    args = SimpleArgumentParser()
    if overriding_args is not None:
        args.parse_args(overriding_args)
    else:
        args.parse_args()

    set_seed()
    set_sharing_strategy()
    device = get_device()

    if args.rtpt_enabled:
        rtpt = RTPT(name_initials='mwache',
                    experiment_name='explainer-critic',
                    max_iterations=args.n_iterations)
    else:
        rtpt = None

    if args.logging_disabled or eval_mode:
        logging = None
        writer = None
    else:
        log_dir = f"./runs/{config_string(args)}"
        write_config_to_log(args, log_dir)
        writer = SummaryWriter(log_dir)
        logging = Logging(writer, args.run_name, args.log_interval, args.log_interval_accuracy, args.n_test_batches,
                          args.log_interval_critic)

    # image_handler = ImageHandler(device, writer) # TODO: make ImageHandler a singleton
    ImageHandler.device, ImageHandler.writer = device, writer

    return args, device, logging, rtpt


def smooth_end_losses(losses: List[float]) -> float:
    """average the last quarter of the losses"""
    last_few_losses = losses[-len(losses) // 4:len(losses)]
    if last_few_losses:
        return mean(last_few_losses)
    else:
        print("not enough losses to smooth")
        return losses[-1]
