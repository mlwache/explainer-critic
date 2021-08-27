
from config import Config
from net import Net
from typing import Any, List, Tuple
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch import Tensor


class Critic:
    classifier: Net

    def __init__(self, cfg: Config):
        self.classifier: Net = Net(accepts_additional_explanations=False, cfg=cfg)
        self.classifier = self.classifier.to(cfg.DEVICE)
        self.cfg: Config = cfg

    def reset(self):
        """Resets to a new un-trained classifier."""
        self.classifier = Net(accepts_additional_explanations=False, cfg=self.cfg)

    def train(self, critic_loader: DataLoader[Any], explanations: List[Tensor],
              n_explainer_batch: int, use_explanations: bool = True) -> Tuple[float, float]:
        # todo: outsource to Net
        critic_loss: Module = self.cfg.LOSS
        optimizer: Optimizer = self.cfg.optimizer(self.classifier.parameters())

        end_of_training_loss: float = 0.0
        data: list
        initial_loss: float = 0.0
        for n_current_batch, data in enumerate(critic_loader):

            assert n_current_batch <= self.cfg.n_critic_batches

            # get the inputs; data is a list of [inputs, labels]
            inputs: Tensor
            inputs, labels = data
            inputs = inputs.to(self.cfg.DEVICE)
            labels = labels.to(self.cfg.DEVICE)

            # zero the parameter gradients # TODO: think about whether zero_grad should be done here.
            optimizer.zero_grad()

            # forward + backward + optimize
            if use_explanations:
                assert inputs.size() == explanations[n_current_batch].size()
            input_explanation_product: Tensor = inputs * explanations[n_current_batch] if use_explanations else inputs
            assert inputs.size() == input_explanation_product.size()  # element-wise product should not change the size.
            outputs = self.classifier(input_explanation_product)
            loss = critic_loss(outputs, labels)
            if n_current_batch == 0:
                initial_loss = loss.item()
            loss.backward()  # or retain_graph=True ? TODO: clarify computation graph
            optimizer.step()

            # print statistics
            print(f'critic n_current_batch = {n_current_batch}, loss.item() = {loss.item():.3f}')
            global_step = self.cfg.n_critic_batches*n_explainer_batch + n_current_batch
            if hasattr(self.cfg, "WRITER"):
                self.cfg.WRITER.add_scalar("Critic_Training/Critic_Loss", loss.item(), global_step=global_step)
            end_of_training_loss = loss.item()  # TODO: think about computation graph

        return initial_loss, end_of_training_loss
