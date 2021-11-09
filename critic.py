from statistics import mean
from typing import Any, List, Tuple, Optional

from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import global_vars
from net import Net

Loss = float


class Critic:

    def __init__(self, device: str, critic_loader: DataLoader[Any], writer: Optional[SummaryWriter],
                 log_interval_critic: Optional[int]):
        self.classifier = Net().to(device)
        self.critic_loader = critic_loader
        self.writer = writer
        self.log_interval_critic = log_interval_critic

    def train(self, explanations: List[Tensor], critic_learning_rate: float) -> Tuple[float, float, float]:

        self.classifier.train()

        critic_loss: Module = nn.CrossEntropyLoss()
        optimizer: Optimizer = optim.Adadelta(self.classifier.parameters(), lr=critic_learning_rate)

        losses: List[float] = []

        for n_current_batch, (inputs, labels) in enumerate(self.critic_loader):
            losses.append(self._process_batch(critic_loss, explanations,
                                              inputs, labels, n_current_batch,
                                              optimizer))
            global_vars.global_step += 1

        return losses[0], losses[-1], mean(losses)

    def _process_batch(self, loss_function: nn.Module, explanations: List[Tensor], inputs: Tensor, labels: Tensor,
                       n_current_batch: int, optimizer) -> Loss:

        optimizer.zero_grad()

        if explanations:
            input_explanation_product: Tensor = inputs * explanations[n_current_batch]
        else:
            input_explanation_product = inputs
        outputs = self.classifier(input_explanation_product)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        self._log_results(loss, n_current_batch)
        return loss.item()

    def _log_results(self, loss, n_current_batch):
        if self.log_interval_critic and n_current_batch % self.log_interval_critic == 0:
            # if n_current_batch == 0 or n_current_batch == self.cfg.n_critic_batches - 1:

            print(f'crit_batch = {n_current_batch}, loss.item() = {loss.item():.3f}')
            if self.writer:
                self.add_scalars_to_writer(loss)

    def add_scalars_to_writer(self, loss):
        # global_step = self.cfg.n_critic_batches * n_explainer_batch + n_current_batch
        if self.writer:
            self.writer.add_scalar("Critic_Training/Critic_Loss", loss.item(), global_step=global_vars.global_step)
