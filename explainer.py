from captum.attr import InputXGradient
import torch

from torch.utils.data.dataloader import DataLoader
from torch import Tensor
from typing import Any, Tuple
from torch.optim import Optimizer

from torch.nn.modules import Module
from config import Config
from critic import Critic

from net import Net


class Explainer:
    classifier: Net

    def __init__(self, cfg: Config):
        self.classifier = Net(accepts_additional_explanations=False, cfg=cfg)
        self.classifier = self.classifier.to(cfg.DEVICE)
        self.critic = Critic(cfg)

        self.cfg = cfg

    def compute_accuracy(self, test_loader: DataLoader[Any]):
        n_correct_samples: int = 0
        n_test_samples_total: int = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                assert i <= self.cfg.n_test_batches
                labels: Tensor
                images, labels = data
                images = images.to(self.cfg.DEVICE)
                labels = labels.to(self.cfg.DEVICE)

                # calculate outputs by running images through the network
                outputs = self.classifier(images)
                assert outputs.device.type == self.cfg.DEVICE

                # the class with the highest output is what we choose as prediction
                predicted: Tensor
                _, predicted = torch.max(outputs.data, dim=1)
                n_test_samples_total += labels.size()[0]
                n_correct_samples += (predicted == labels).sum().item()
                assert predicted.device.type == self.cfg.DEVICE
        total_accuracy = n_correct_samples / n_test_samples_total
        assert n_test_samples_total == self.cfg.n_test_samples
        return total_accuracy

    def save_model(self):
        torch.save(self.classifier.state_dict(), self.cfg.PATH_TO_MODELS)

    def train(self, train_loader: DataLoader[Any], critic_loader: DataLoader[Any] = None,
              use_critic: bool = True) -> Tuple[float, float]:
        # check Argument validity
        assert not (use_critic and critic_loader is None)
        # assure the critic isn't used
        if not use_critic:
            critic_loader = None

        classification_loss: Module = self.cfg.LOSS
        # actually the type is _Loss, but that's protected, for backward compatibility.
        # https://discuss.pytorch.org/t/why-is-the-pytorch-loss-base-class-protected/123417
        optimizer: Optimizer = self.cfg.optimizer(self.classifier.parameters())

        initial_loss: float = 0.0
        end_of_training_loss: float = 0.0

        for epoch in range(self.cfg.n_epochs):
            loss_explanation: float = 0.0
            for n_current_batch, data in enumerate(train_loader):

                assert n_current_batch < self.cfg.n_training_batches
                if use_critic:
                    self.critic.reset()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(self.cfg.DEVICE)
                labels = labels.to(self.cfg.DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier(inputs)
                assert outputs.device.type == self.cfg.DEVICE
                loss_classification = classification_loss(outputs, labels)
                if use_critic:
                    loss_explanation = self.explanation_loss(critic_loader, n_current_batch)
                    loss = loss_classification + loss_explanation
                else:
                    loss = loss_classification
                if n_current_batch == 0:
                    initial_loss = loss.item()
                loss.backward()
                optimizer.step()

                self.add_scalars_to_writer(loss, loss_classification, loss_explanation, n_current_batch)

                self.print_statistics(loss, loss_classification, loss_explanation, n_current_batch)

                end_of_training_loss = loss.item()

        self.terminate_writer()

        return initial_loss, end_of_training_loss

    def terminate_writer(self):
        if hasattr(self.cfg, "WRITER"):
            self.cfg.WRITER.flush()
            self.cfg.WRITER.close()

    def print_statistics(self, loss, loss_classification, loss_explanation, n_current_batch):
        # print statistics
        print(f'explainer [batch  {n_current_batch}] \n'
              f'Loss: {loss:.3f} = {loss_classification:.3f}(classification)'
              f' + {loss_explanation:.3f}(explanation)')
        if self.cfg.rtpt_enabled:
            self.cfg.RTPT_OBJECT.step(subtitle=f"loss={loss:2.2f}")

    def add_scalars_to_writer(self, loss, loss_classification, loss_explanation, n_current_batch):

        global_step = n_current_batch * self.cfg.n_critic_batches if self.cfg.n_critic_batches != 0 else n_current_batch

        if hasattr(self.cfg, "WRITER"):
            self.cfg.WRITER.add_scalar("Explainer_Training/Explanation", loss_explanation,
                                       global_step=global_step)
            self.cfg.WRITER.add_scalar("Explainer_Training/Classification", loss_classification,
                                       global_step=global_step)
            self.cfg.WRITER.add_scalar("Explainer_Training/Total", loss, global_step=global_step)

    def explanation_loss(self, critic_loader: DataLoader, n_current_batch: int) -> float:
        explanations = []
        for inputs, labels in critic_loader:
            inputs = inputs.to(self.cfg.DEVICE)
            labels = labels.to(self.cfg.DEVICE)
            explanations.append(self.input_gradient(inputs, labels))

        critic_end_of_training_loss: float
        _, critic_end_of_training_loss = self.critic.train(critic_loader, explanations, n_current_batch)
        return critic_end_of_training_loss

    def input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        assert input_images.size() == torch.Size([self.cfg.batch_size, 1, 28, 28])
        assert labels.size() == torch.Size([self.cfg.batch_size])
        input_x_gradient = InputXGradient(self.classifier.forward)
        input_images.requires_grad = True
        gradient_x_input_one_image: Tensor = input_x_gradient.attribute(inputs=input_images, target=labels)
        gradient: Tensor = gradient_x_input_one_image / input_images

        # The gradient tensor is computed from other tensors on cfg.device, so it should be there.
        assert gradient.device.type == self.cfg.DEVICE
        return gradient

    def predict(self, images: Tensor) -> Tensor:
        outputs = self.classifier(images)
        _, prediction = torch.max(outputs, 1)
        return prediction
