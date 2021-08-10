import torch.nn.functional as f
import torch.nn as nn
from torch import Tensor
import warnings
import torch
from config import default_config as cfg


class Net(nn.Module):
    def __init__(self, accepts_additional_explanations: bool):
        super().__init__()
        self.accepts_additional_explanations = accepts_additional_explanations
        if accepts_additional_explanations:
            # noinspection PyTypeChecker
            self.conv1 = nn.Conv3d(1, 10, kernel_size=5)  # TODO: check which dimensions we need here.
            raise NotImplementedError  # this part is still work in progress.
        else:
            # noinspection PyTypeChecker
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        if self.accepts_additional_explanations:
            print(x.size(), " x.size()")
            assert x.size() == torch.Size([cfg.batch_size, 2, 1, 28, 28])
        else:
            assert x.size() == torch.Size([cfg.batch_size, 1, 28, 28])
        with warnings.catch_warnings():  # ignore the named tensor warning as it's not important,
            # and it adds visual clutter. (It's not important because I will keep the venv stable,
            # and my code is not critical for anyone. Warning Text:
            # "UserWarning: Named tensors and all their associated APIs are an experimental feature
            # and subject to change. Please do not use them for anything important
            # until they are released as stable."
            warnings.simplefilter("ignore")
            x = f.relu(
                f.max_pool2d(
                    self.conv1(x),
                    2)
            )
            x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=-1)
        assert x.size() == torch.Size([cfg.batch_size, len(cfg.CLASSES)])
        return x   # Implicit dimension choice for log_softmax
        # has been deprecated. Just using the last dimension for now.
        # (https://stackoverflow.com/questions/49006773/userwarning-implicit-dimension-choice-for-log-softmax-has-been-deprecated)
