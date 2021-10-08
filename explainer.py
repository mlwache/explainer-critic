from typing import Any, Tuple, List, Optional

import torch
from captum.attr import InputXGradient
from rtpt import RTPT
from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import SimpleArgumentParser
from critic import Critic
from learner import Learner
from net import Net
from visualization import ImageHandler

Loss = float


class Explainer(Learner):
    classifier: Net

    def __init__(self, cfg: SimpleArgumentParser, device: str, writer: SummaryWriter = None):
        super().__init__(cfg, device, writer)
        self.rtpt: Optional[RTPT] = None
        self.writer_step_offset: int = 0

    def start_rtpt(self, n_training_batches):
        self.rtpt = RTPT(name_initials='mwache',
                         experiment_name='explainer-critic',
                         max_iterations=n_training_batches)
        self.rtpt.start()

    def compute_accuracy(self, test_loader: DataLoader[Any]):
        n_correct_samples: int = 0
        n_test_samples_total: int = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                assert i <= self.cfg.n_test_samples
                labels: Tensor
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.classifier(images)
                assert outputs.device.type == self.device

                # the class with the highest output is what we choose as prediction
                predicted: Tensor
                _, predicted = torch.max(outputs.data, dim=1)
                n_test_samples_total += labels.size()[0]
                n_correct_samples += (predicted == labels).sum().item()
                assert predicted.device.type == self.device
        total_accuracy = n_correct_samples / n_test_samples_total
        assert n_test_samples_total == self.cfg.n_test_samples
        return total_accuracy

    def save_model(self):
        torch.save(self.classifier.state_dict(), './models/mnist_net.pth')

    def pre_train(self, train_loader: DataLoader[Any], n_epochs: int = -1) -> Tuple[Loss, Loss]:
        if n_epochs == -1:
            n_epochs = self.cfg.n_pretraining_epochs
        return self.train(train_loader, use_critic=False, n_epochs=n_epochs)

    def update_epoch_writer_step_offset(self, train_loader: DataLoader[Any]):
        # if critic_loader:
        #     critic_set_size = len(critic_loader)
        # else:
        #     critic_set_size = 1
        training_set_size = len(train_loader)
        self.writer_step_offset += training_set_size * self.cfg.n_critic_batches

    def train(self, train_loader: DataLoader[Any], critic_loader: DataLoader[Any] = None,
              use_critic: bool = True, n_epochs: int = 0) -> Tuple[Loss, Loss]:
        # check Argument validity
        assert not (use_critic and critic_loader is None)
        optimizer: Optimizer = optim.Adadelta(self.classifier.parameters(), lr=self.cfg.learning_rate_start)
        if not use_critic:  # pretraining mode
            critic_loader = None
            for g in optimizer.param_groups:  # To do: tidy this up.
                g['lr'] = self.cfg.pretrain_learning_rate
        if n_epochs == 0:  # if n_epochs isn't set
            n_epochs = self.cfg.n_epochs
            for g in optimizer.param_groups:
                g['lr'] = self.cfg.learning_rate_start
        loss_function_classification: Module = nn.CrossEntropyLoss()
        # actually the type is _Loss, but that's protected, for backward compatibility.
        # https://discuss.pytorch.org/t/why-is-the-pytorch-loss-base-class-protected/123417

        losses: List[float] = []
        # scheduler = StepLR(optimizer, step_size=1, gamma=self.cfg.learning_rate_step)
        for current_epoch in range(n_epochs):
            print(f"[epoch {current_epoch}]")
            for n_current_batch, (inputs, labels) in enumerate(train_loader):
                losses.append(self._process_batch(loss_function_classification, inputs, labels,
                                                  n_current_batch, optimizer, critic_loader=critic_loader))
            self.update_epoch_writer_step_offset(train_loader)
            # scheduler.step()
        self.terminate_writer()

        return losses[0], super().smooth_end_losses(losses)

    def _process_batch(self, loss_function: nn.Module, inputs: Tensor, labels: Tensor,
                       n_current_batch: int, optimizer: Optimizer,
                       # n_explainer_batch: int = 0, explanations: List[Tensor] = None,
                       critic_loader: DataLoader[Any] = None) -> Loss:
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.classifier(inputs)
        loss_classification = loss_function(outputs, labels)
        loss = self._add_explanation_loss(critic_loader, loss_classification, n_current_batch)
        loss.backward()
        optimizer.step()
        self._sanity_check_batch_device(n_current_batch, outputs)
        self._record_losses(loss, loss_classification, n_current_batch)
        return loss.item()

    def _sanity_check_batch_device(self, n_current_batch, outputs):
        assert n_current_batch < self.cfg.n_training_batches
        assert outputs.device.type == self.device

    def _record_losses(self, loss, loss_classification, n_current_batch):
        self.add_scalars_to_writer(loss, loss_classification, loss - loss_classification, n_current_batch)
        self.print_statistics(loss, loss_classification, loss - loss_classification, n_current_batch)

    def _add_explanation_loss(self, critic_loader, loss_classification, n_current_batch):
        if critic_loader:
            loss_explanation = self.explanation_loss(critic_loader, n_current_batch)
            loss = loss_classification + loss_explanation
        else:
            loss = loss_classification
        return loss

    def terminate_writer(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def print_statistics(self, loss, loss_classification, loss_explanation, n_current_batch):
        # print statistics
        print(f'explainer [batch  {n_current_batch}] \n'
              f'Loss: {loss:.3f} = {loss_classification:.3f}(classification)'
              f' + {loss_explanation:.3f}(explanation)')
        if self.cfg.rtpt_enabled:
            self.rtpt.step(subtitle=f"loss={loss:2.2f}")

    def add_scalars_to_writer(self, loss, loss_classification, loss_explanation, n_current_batch):

        relative_step = n_current_batch * self.cfg.n_critic_batches if \
            self.cfg.n_critic_batches != 0 else n_current_batch
        global_step = self.writer_step_offset + relative_step

        if self.writer:
            self.writer.add_scalar("Explainer_Training/Explanation", loss_explanation,
                                   global_step=global_step)
            self.writer.add_scalar("Explainer_Training/Classification", loss_classification,
                                   global_step=global_step)
            self.writer.add_scalar("Explainer_Training/Total", loss, global_step=global_step)

    def explanation_loss(self, critic_loader: DataLoader, n_current_batch: int) -> float:
        critic = Critic(self.cfg, self.device, self.writer)
        explanations = []
        for inputs, labels in critic_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations.append(self.rescaled_input_gradient(inputs, labels))

        critic_end_of_training_loss: float
        n_current_batch_total = (self.writer_step_offset // len(critic_loader)) + n_current_batch
        _, critic_end_of_training_loss = critic.train(critic_loader, explanations, n_current_batch_total)

        return critic_end_of_training_loss

    def input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        assert input_images.size()[1:] == torch.Size([1, 28, 28])
        # usually but not always: torch.Size([self.cfg.batch_size, 1, 28, 28])
        # assert labels.size() == torch.Size([self.cfg.batch_size])
        input_x_gradient = InputXGradient(self.classifier.forward)
        input_images.requires_grad = True
        gradient_x_input_one_image: Tensor = input_x_gradient.attribute(inputs=input_images, target=labels)
        gradient: Tensor = gradient_x_input_one_image / input_images

        # The gradient tensor is computed from other tensors on cfg.device, so it should be there.
        assert gradient.device.type == self.device
        return gradient

    def rescaled_input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        gradient = self.input_gradient(input_images, labels)
        return ImageHandler.re_scale_to_zero_one(gradient)

    def predict(self, images: Tensor) -> Tensor:
        outputs = self.classifier(images)
        _, prediction = torch.max(outputs, 1)
        return prediction
