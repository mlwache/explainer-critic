import torch.nn.functional as f
import torch.nn as nn
from torch import Tensor
import torch


class Net(nn.Module):
    # Use the same architecture as https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, cfg, accepts_additional_explanations: bool = False):
        super().__init__()
        self.cfg = cfg
        self.accepts_additional_explanations = accepts_additional_explanations
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output
