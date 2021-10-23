from statistics import mean
from typing import List, Optional, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import global_vars
from config import SimpleArgumentParser
from net import Net
from utils import colored

Loss = float


class Learner:

    def __init__(self, cfg: SimpleArgumentParser, device: str, writer: Optional[SummaryWriter]):
        classifier: Net = Net(cfg=cfg)
        self.cfg: SimpleArgumentParser = cfg
        self.device: str = device
        self.writer: Optional[SummaryWriter] = writer

        self.classifier = classifier.to(device)

    # noinspection PyMethodMayBeStatic
    def _smooth_end_losses(self, losses: List[float]) -> Loss:
        """average the last quarter of the losses"""
        last_few_losses = losses[-len(losses) // 4:len(losses)]
        if last_few_losses:
            return mean(last_few_losses)
        else:
            print("not enough losses to smooth")
            return losses[-1]

    def reset(self):
        """Resets to a new un-trained classifier."""
        self.classifier = Net(cfg=self.cfg)

    def compute_accuracy(self, data_loader: DataLoader[Any]):
        n_correct_samples: int = 0
        n_test_samples_total: int = 0

        self.classifier.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if i >= self.cfg.n_test_batches:  # only test on a set of the test set size, even for training accuracy.
                    break
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.classifier(images)

                # the class with the highest output is what we choose as prediction
                _, predicted = torch.max(outputs.data, dim=1)
                n_test_samples_total += labels.size()[0]
                n_correct_samples += (predicted == labels).sum().item()
                assert predicted.device.type == self.device
        total_accuracy = n_correct_samples / n_test_samples_total
        # assert n_test_samples_total == self.cfg.n_test_samples
        return total_accuracy

    def log_accuracy(self, train_loader: DataLoader[Any], test_loader: DataLoader[Any], n_current_batch: int):
        global_step = global_vars.global_step
        training_accuracy = self.compute_accuracy(train_loader)
        # test_accuracy = -0.1  # just for initializing. negative so that we will notice if it's unchanged
        test_accuracy = self.compute_accuracy(test_loader)
        print(colored(0, 0, 200, f'accuracy training: {training_accuracy}, accuracy testing: {test_accuracy:.3f}'))
        if self.writer:
            self.writer.add_scalar("Explainer_Training/Training_Accuracy", training_accuracy, global_step=global_step)
            if test_loader:
                self.writer.add_scalar("Explainer_Training/Test_Accuracy", test_accuracy, global_step=global_step)

    # def global_step(self, n_current_batch: int) -> int:
    #     # maybe to do: make this an abstract base class. for now just throw an error
    #     raise ValueError("Learner's global_step should not be called.")

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
