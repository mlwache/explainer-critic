import torch
from torch import Tensor

from torch.utils.data import DataLoader
from typing import Any
import main
from config import SimpleArgumentParser


def test_main_load_data():
    args = SimpleArgumentParser()
    args.parse_args()
    train_loader: DataLoader[Any]
    test_loader: DataLoader[Any]
    critic_loader: DataLoader[Any]
    train_loader, test_loader, critic_loader = main.load_data(args)
    train_data_sample: Tensor
    for i, train_data_sample in enumerate(train_loader):
        assert i <= args.n_training_batches
        images, labels = train_data_sample
        assert len(images) == args.batch_size
        assert images[0].size() == torch.Size([1, 28, 28])
        assert torch.all(images[0].data.ge(-1*torch.ones_like(images)))
        # the upper bound is higher, as the images are normalized and there are less white/high-value pixels
        assert torch.all(images[0].data.le(5*torch.ones_like(images)))
