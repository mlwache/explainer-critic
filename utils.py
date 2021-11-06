import json
import os
import random
from datetime import datetime
from typing import Tuple, Any

import numpy as np
import torch.cuda
import torch.multiprocessing
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torchvision.datasets import MNIST

from config import SimpleArgumentParser


def load_data_from_args(args: SimpleArgumentParser) -> Tuple[DataLoader[Any], DataLoader[Any],
                                                             DataLoader[Any]]:
    return load_data(args.n_training_samples, args.n_critic_samples, args.n_test_samples, args.batch_size)


# noinspection PyShadowingNames
def load_data(n_training_samples: int, n_critic_samples: int, n_test_samples: int,
              batch_size: int) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:

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
    test_set, _ = random_split(full_test_set, test_split)
    test_set = test_set.dataset
    # for the visualization get 50 samples of the dataset, 5 for each label
    # visualization_sets = []
    # for label in range(10):
    #     visualization_sets.append(Subset(test_set, np.where(test_set.targets == label)[0][:5]))
    # visualization_set = ConcatDataset(visualization_sets)

    train_loader: DataLoader[Any] = DataLoader(training_set, batch_size=batch_size, num_workers=0)
    critic_loader: DataLoader[Any] = DataLoader(critic_set, batch_size=batch_size, num_workers=0)
    test_loader: DataLoader[Any] = DataLoader(test_set, batch_size=batch_size, num_workers=0)
    # visualization_loader: DataLoader[Any] = DataLoader(visualization_set, batch_size=batch_size, num_workers=0)

    return train_loader, test_loader, critic_loader


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
    lr_mode = "const_lr" if cfg.constant_lr else "sched"

    # just for somewhat nicer formatting:
    run_name = cfg.run_name + "_" if cfg.run_name else ""
    
    return f'{run_name}' \
           f'{cfg.training_mode}_ex{cfg.n_training_batches}_cr{cfg.n_critic_batches}' \
           f'_lr{cfg.learning_rate_start}_{lr_mode}' \
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        mean_mnist = 0.1307
        std_dev_mnist = 0.3081

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(mean_mnist).div_(std_dev_mnist)

        # Put both data and targets on GPU in advance
        device = get_device()
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
