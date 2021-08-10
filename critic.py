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

    def train(self, critic_loader: DataLoader[Any], explanations: List[Tensor]) -> float:
        # todo: outsource to Net
        critic_loss: Module = cfg.LOSS
        optimizer: Optimizer = cfg.optimizer(self.classifier.parameters())
        intermediate_loss_sum = 0.0
        intermediate_loss_count = 0
        end_of_training_loss: float = 0.0
        data: list
        for n_current_batch, data in enumerate(critic_loader, 0):  # i is the index of the current batch.

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
            intermediate_loss_sum += loss.item()
            intermediate_loss_count += 1
            if cfg.n_critic_batches <= cfg.TIMES_TO_PRINT_CRITIC or (
                    n_current_batch % (cfg.n_critic_batches // cfg.TIMES_TO_PRINT_CRITIC) == 0
            ):
                print('critic[batch %5d] loss: %.3f' %
                      (n_current_batch, intermediate_loss_sum / intermediate_loss_count))
                # (average over the last part of the batches)
                intermediate_loss_sum = 0.0
            end_of_training_loss = loss.item()  # or just loss? TODO: think about computation graph

        return end_of_training_loss
