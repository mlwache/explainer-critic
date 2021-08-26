from torch.utils.tensorboard import SummaryWriter

from net import Net
from typing import Any, List
from torch.utils.data.dataloader import DataLoader
from config import default_config as cfg
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch import Tensor


class Critic:
    classifier: Net

    def __init__(self):
        self.classifier = Net(accepts_additional_explanations=False)
        self.classifier = self.classifier.to(cfg.DEVICE)

    def train(self, critic_loader: DataLoader[Any], explanations: List[Tensor], writer: SummaryWriter, n_explainer_batch: int) -> float:
        # todo: outsource to Net
        critic_loss: Module = cfg.LOSS
        optimizer: Optimizer = cfg.optimizer(self.classifier.parameters())

        end_of_training_loss: float = 0.0
        data: list
        for n_current_batch, data in enumerate(critic_loader):  # i is the index of the current batch.

            # only train on a part of the samples.
            if n_current_batch >= cfg.n_critic_batches:
                break

            # get the inputs; data is a list of [inputs, labels]
            inputs: Tensor
            inputs, labels = data
            inputs = inputs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            # zero the parameter gradients # TODO: think about whether zero_grad should be done here.
            # optimizer.zero_grad()

            # forward + backward + optimize
            assert inputs.size() == explanations[n_current_batch].size()
            input_explanation_product: Tensor = inputs * explanations[n_current_batch]
            assert inputs.size() == input_explanation_product.size()  # element-wise product should not change the size.
            outputs = self.classifier(input_explanation_product)
            loss = critic_loss(outputs, labels)
            loss.backward()  # or retain_graph=True ? TODO: clarify computation graph
            optimizer.step()

            # print statistics
            print(f'critic n_current_batch = {n_current_batch}, loss.item() = {loss.item():.3f}')
            global_step = cfg.n_critic_batches*n_explainer_batch+n_current_batch
            writer.add_scalar("Critic_Training/Critic_Loss", loss.item(), global_step=global_step)
            end_of_training_loss = loss.item()  # TODO: think about computation graph

        return end_of_training_loss
