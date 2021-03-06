import json
import os
import random
from datetime import datetime
from typing import Tuple, Any, Optional, List, Union

import git
import numpy as np
import torch.cuda
import torch.multiprocessing
from torch import Tensor, nn
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

from config import SimpleArgumentParser
from helper_types import Loaders, Logging
import global_vars


def load_data_from_args(args: SimpleArgumentParser) -> Loaders:
    return load_data(n_training_samples=args.n_training_samples,
                     n_critic_samples=args.n_critic_samples,
                     n_test_samples=args.n_test_samples,
                     batch_size=args.batch_size,
                     test_batch_size=args.test_batch_size)


def get_test_loader(n_samples: int, batch_size: int, random_permutation: bool = True) -> DataLoader:
    full_test_set = FastMNIST('./data', train=False, download=True)
    if random_permutation:
        test_split = [n_samples, len(full_test_set) - n_samples]
        test_set, _ = random_split(full_test_set, test_split)
        return DataLoader(test_set, batch_size=batch_size)
    else:
        visualization_sets = []
        for label in range(10):
            visualization_sets.append(Subset(full_test_set,
                                             torch.where(full_test_set.targets == label)[0][:n_samples // 10]))
        visualization_set = ConcatDataset(visualization_sets)
        return DataLoader(visualization_set, batch_size=batch_size)


# noinspection PyShadowingNames
def load_data(n_training_samples: int,
              n_critic_samples: int,
              n_test_samples: int,
              batch_size: int,
              test_batch_size: int) -> Loaders:
    training_and_critic_set = FastMNIST('./data', train=True, download=True)
    # loads the data to the ./data folder

    # check that we have enough samples:
    n_total_training_samples = n_training_samples + n_critic_samples
    n_spare_samples = len(training_and_critic_set) - n_total_training_samples
    assert n_spare_samples >= 0, f"{n_total_training_samples} samples are too much. " \
                                 f"Please reduce the number of training or critic batches, or the batch size."

    # split training set into one training set for the classification, and one for the critic
    train_split = [n_training_samples, n_critic_samples, n_spare_samples]
    training_set, critic_set, _ = random_split(training_and_critic_set, train_split)

    loaders = Loaders(
        train=DataLoader(training_set, batch_size=batch_size, num_workers=0, shuffle=True),
        # for the critic set, I do the shuffling explicitly during training
        # in order to match samples to their respective explanations.
        critic=DataLoader(critic_set, batch_size=batch_size, num_workers=0),
        test=get_test_loader(n_test_samples, test_batch_size),
        visualization=get_test_loader(n_samples=40, batch_size=40, random_permutation=False))

    return loaders


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def set_device():
    if not torch.cuda.is_available():
        print(colored(200, 150, 0, f"No GPU found, falling back to CPU."))
        global_vars.DEVICE = "cpu"
    else:
        global_vars.DEVICE = "cuda"


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


def get_one_batch_of_images(loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
    images, labels = next(iter(loader))
    images, labels = images.to(global_vars.DEVICE), labels.to(global_vars.DEVICE)
    return images, labels


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args(overriding_args: Optional[List]) -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    if overriding_args is not None:
        args.parse_args(overriding_args)
    else:
        args.parse_args()
    return args


def setup(args: SimpleArgumentParser, eval_mode: bool = False) -> None:
    set_seed(args.random_seed)
    set_sharing_strategy()
    set_device()

    if args.logging_disabled or eval_mode:
        global_vars.LOGGING = None
    else:
        log_dir = f"./runs/{config_string(args)}"
        write_config_to_log(args, log_dir)
        writer = SummaryWriter(log_dir)
        global_vars.LOGGING = Logging(writer, args.run_name, args.log_interval, args.log_interval_accuracy,
                                      args.n_test_batches, args.log_interval_critic)


def get_git_root() -> str:
    current_path = os.path.dirname(os.path.realpath(__file__))
    git_repo = git.Repo(current_path, search_parent_directories=True)
    return git_repo.git.rev_parse("--show-toplevel")


def get_data_tensors(n_samples: int) -> Tuple[Tensor, Tensor]:
    return loader_to_tensors(get_test_loader(n_samples=n_samples, batch_size=n_samples))


def compute_accuracy(classifier: nn.Module,
                     data: Union[DataLoader, List[List[Tensor]]],
                     n_batches: Optional[int] = None):
    if n_batches is None:
        n_batches = len(data)
    n_correct_samples: int = 0
    n_test_samples_total: int = 0

    classifier.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(data):
            if i >= n_batches:  # only test on a set of the test set size, even for training accuracy.
                break

            outputs = classifier(images)

            # the class with the highest output is what we choose as prediction
            _, predicted = torch.max(outputs.data, dim=1)
            n_test_samples_total += labels.size()[0]
            n_correct_samples += (predicted == labels).sum().item()
    total_accuracy = n_correct_samples / n_test_samples_total
    classifier.train()
    return total_accuracy


def loader_to_tensors(dataloader: DataLoader) -> Tuple[Tensor, Tensor]:
    all_input_batches = []
    all_label_batches = []
    for input_batch, label_batch in dataloader:
        all_input_batches.append(input_batch)
        all_label_batches.append(label_batch)
    input_tensor = torch.flatten(torch.stack(all_input_batches), start_dim=0, end_dim=1)
    label_tensor = torch.flatten(torch.stack(all_label_batches), start_dim=0, end_dim=1)
    return input_tensor, label_tensor


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
        self.data, self.targets = self.data.to(global_vars.DEVICE), self.targets.to(global_vars.DEVICE)

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
