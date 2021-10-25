import json
import os
import random
import warnings
from datetime import datetime
from typing import Tuple, Any

import numpy as np
import torch.cuda
import torch.multiprocessing
import torch.utils
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import MNIST

from config import SimpleArgumentParser


def load_data_from_args(args: SimpleArgumentParser) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    return load_data(args.n_training_samples, args.n_critic_samples, args.n_test_samples, args.batch_size)


# noinspection PyShadowingNames
def load_data(n_training_samples: int, n_critic_samples: int, n_test_samples: int, batch_size: int) -> \
        Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
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
        # loads the data to .data folder
        # ignores the UserWarning: The given NumPy array is not writeable,
        # and PyTorch does not support non-writeable tensors.
        # This means you can write to the underlying (supposedly non-writeable)
        # NumPy array using the tensor. You may want to copy the array to protect its data
        # or make it writeable before converting it to a tensor.
        # This type of warning will be suppressed for the rest of this program.

    n_total_training_samples = n_training_samples + n_critic_samples
    n_spare_samples = len(training_and_critic_set) - n_total_training_samples
    assert n_spare_samples >= 0, f"{n_total_training_samples} are too much."
    split = [n_training_samples, n_critic_samples, n_spare_samples]
    training_set, critic_set, _ = torch.utils.data.random_split(training_and_critic_set, split)

    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                                                num_workers=0)
    critic_loader: DataLoader[Any] = torch.utils.data.DataLoader(critic_set, batch_size=batch_size,
                                                                 num_workers=0)

    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    split = [n_test_samples, len(test_set) - n_test_samples]
    test_set, _ = torch.utils.data.random_split(test_set, split)
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                               num_workers=0)
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
