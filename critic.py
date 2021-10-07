from torch.utils.tensorboard import SummaryWriter

from config import SimpleArgumentParser
from learner import Learner
from typing import Any, List, Tuple, Optional
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch import Tensor, nn, optim

Loss = float


class Critic(Learner):

    def __init__(self, cfg: SimpleArgumentParser, device: str, writer: Optional[SummaryWriter] = None):
        super().__init__(cfg, device, writer)

    def train(self, critic_loader: DataLoader[Any], explanations: List[Tensor],
              n_explainer_batch: int) -> Tuple[float, float]:
        critic_loss: Module = self.cfg.LOSS
        optimizer: Optimizer = optim.Adadelta(self.classifier.parameters(), lr=self.cfg.learning_rate_start)

        losses: List[float] = []
        for n_current_batch, (inputs, labels) in enumerate(critic_loader):
            losses.append(self._process_batch(critic_loss, explanations,
                                              inputs, labels, n_current_batch,
                                              n_explainer_batch, optimizer))

        return losses[0], super().smooth_end_losses(losses)

    def _process_batch(self, loss_function: nn.Module, explanations: List[Tensor], inputs: Tensor, labels: Tensor,
                       n_current_batch: int, n_explainer_batch: int, optimizer) -> Loss:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # zero the parameter gradients # TODO: think about whether zero_grad should be done here.
        optimizer.zero_grad()
        # forward + backward + optimize
        if explanations:
            input_explanation_product: Tensor = inputs * explanations[n_current_batch]
        else:
            input_explanation_product = inputs
        outputs = self.classifier(input_explanation_product)
        loss = loss_function(outputs, labels)
        loss.backward()  # or retain_graph=True ? TODO: clarify computation graph
        optimizer.step()
        self._sanity_check(input_explanation_product, inputs, n_current_batch)
        self._log_results(loss, n_current_batch, n_explainer_batch)
        return loss.item()

    def _sanity_check(self, inputs, input_explanation_product, n_current_batch):
        # sanity checks
        assert inputs.size() == input_explanation_product.size()  # element-wise product should not change the size.
        assert n_current_batch <= self.cfg.n_critic_batches

    def _log_results(self, loss, n_current_batch, n_explainer_batch):
        if n_current_batch == 0:  # only print the beginning for now.
            print(f'critic n_current_batch = {n_current_batch}, loss.item() = {loss.item():.3f}')
        self.add_scalars_to_writer(loss, n_current_batch, n_explainer_batch)

    def add_scalars_to_writer(self, loss, n_current_batch, n_explainer_batch):
        global_step = self.cfg.n_critic_batches * n_explainer_batch + n_current_batch
        if self.writer:
            self.writer.add_scalar("Critic_Training/Critic_Loss", loss.item(), global_step=global_step)
