import json
import os
import warnings
from datetime import datetime
from typing import Tuple, Any

import torch.cuda
import torch.multiprocessing
import torch.utils
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST

from config import SimpleArgumentParser


# noinspection PyShadowingNames
def load_data(cfg) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # dataset splits for the different parts of training and testing
    training_set: MNIST
    critic_set: MNIST
    test_set: MNIST

    mean_mnist = 0.1307
    std_dev_mnist = 0.3081
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(),
         torchvision.transforms.Normalize((mean_mnist,), (std_dev_mnist,))
         ])
    # transformation that first makes data to a tensor, and then normalizes them.
    # I took the mean and stddev from here:
    # https://nextjournal.com/gkoehler/pytorch-mnist (taking them as given for now)
    # maybe to do: compute them myself, that seems more robust than taking magic numbers from the internet.

    with warnings.catch_warnings():  # Ignore warning, as it's caused by the underlying functional,
        # and I think would require me to change the site-packages in order to fix it.
        warnings.simplefilter("ignore")
        training_and_critic_set: MNIST = torchvision.datasets.MNIST('./data', train=True, download=True,
                                                                    transform=transform_mnist)
        max_train_samples = training_and_critic_set.data.size()[0]
        assert_not_too_many_samples(cfg, max_train_samples)
        # loads the data to .data folder
        # ignores the UserWarning: The given NumPy array is not writeable,
        # and PyTorch does not support non-writeable tensors.
        # This means you can write to the underlying (supposedly non-writeable)
        # NumPy array using the tensor. You may want to copy the array to protect its data
        # or make it writeable before converting it to a tensor.
        # This type of warning will be suppressed for the rest of this program.

    n_spare_samples = len(training_and_critic_set) - cfg.n_training_samples - cfg.n_critic_samples
    assert n_spare_samples >= 0
    split = [cfg.n_training_samples, cfg.n_critic_samples, n_spare_samples]
    training_set, critic_set, _ = torch.utils.data.random_split(training_and_critic_set, split)

    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(training_set, batch_size=cfg.batch_size,
                                                                shuffle=True, num_workers=0)
    critic_loader: DataLoader[Any] = torch.utils.data.DataLoader(critic_set, batch_size=cfg.batch_size,
                                                                 shuffle=False, num_workers=0)

    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    split = [cfg.n_test_samples, len(test_set) - cfg.n_test_samples]
    test_set, _ = torch.utils.data.random_split(test_set, split)
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                                               shuffle=True, num_workers=0)
    return train_loader, test_loader, critic_loader


def assert_not_too_many_samples(cfg: SimpleArgumentParser, max_training_samples: int):
    n_total_training_samples = cfg.n_training_samples + cfg.n_critic_samples
    assert n_total_training_samples <= max_training_samples, f"{n_total_training_samples} are too much."


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text} \033[0m"


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


def config_string(cfg: SimpleArgumentParser, additional_string: str = "") -> str:
    date_time: str = str(datetime.now())[0:-7]

    return f'{additional_string}_{cfg.training_mode}_ex{cfg.n_training_batches}_cr{cfg.n_critic_batches}' \
           f'_lr{cfg.learning_rate_start}' \
           f'_bs{cfg.batch_size}_ep{cfg.n_epochs}_p-ep{cfg.n_pretraining_epochs}' \
           f'_gm{cfg.learning_rate_step}_ts{cfg.n_test_batches} {date_time}'
