import os
from typing import Any, Tuple, List, Optional

import torch
from captum.attr import InputXGradient
from rtpt import RTPT
from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

import global_vars
import utils
from config import SimpleArgumentParser
from critic import Critic
from learner import Learner
from net import Net
from utils import colored
from visualization import ImageHandler

Loss = float


class Explainer(Learner):
    classifier: Net
    test_batch_for_visualization: Tuple[Tensor, Tensor]

    def __init__(self, cfg: SimpleArgumentParser, device: str,
                 test_batch_for_visualization: Tuple[Tensor, Tensor] = None, writer: SummaryWriter = None):
        super().__init__(cfg, device, writer)
        self.rtpt: Optional[RTPT] = None
        # self.writer_step_offset: int = 0
        self.test_batch_for_visualization = test_batch_for_visualization
        self.optimizer: Optional[Optimizer] = None
        if cfg.rtpt_enabled:
            self.start_rtpt(cfg.n_iterations)

    def start_rtpt(self, n_iterations):
        self.rtpt = RTPT(name_initials='mwache',
                         experiment_name='explainer-critic',
                         max_iterations=n_iterations)
        self.rtpt.start()

    def load_state(self, path: str):
        checkpoint: dict = torch.load(path)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # currently not needed, as we don't use checkpoints to continue training, only for inference.
        self.classifier.train()

    def save_state(self, path: str, epoch: int = -1, loss: float = -1.0):

        # first rename the previous model file, as torch.save does not necessarily overwrite the old model.
        if os.path.isfile(path):
            os.replace(path, path + "_previous.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(colored(200, 100, 0, f"Saved model to {path}"))

    def pre_train(self, train_loader: DataLoader[Any], test_loader: Optional[DataLoader[Any]],
                  n_epochs: int = -1, log_interval: int = -1) -> Tuple[Loss, Loss]:
        if n_epochs == -1:
            n_epochs = self.cfg.n_pretraining_epochs
        if log_interval == -1:
            log_interval = self.cfg.log_interval
        init_loss, end_loss = self.train(train_loader, critic_loader=None, n_epochs=n_epochs, test_loader=test_loader,
                                         log_interval=log_interval)
        self.save_state('./models/pretrained_model.pt', epoch=-1, loss=end_loss)
        return init_loss, end_loss

    # def update_epoch_writer_step_offset(self, train_loader: DataLoader[Any]):
    #     training_set_size = len(train_loader)
    #     self.writer_step_offset += training_set_size * self.cfg.n_critic_batches

    def train(self, train_loader: DataLoader[Any], critic_loader: Optional[DataLoader[Any]],
              test_loader: Optional[DataLoader[Any]], log_interval: int, n_epochs: int = -1) -> Tuple[Loss, Loss]:
        self.classifier.train()

        if n_epochs == -1:  # if n_epochs isn't explicitly set, use the default.
            n_epochs = self.cfg.n_epochs

        if critic_loader:  # parallel training mode
            self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=self.cfg.learning_rate_start)
        else:  # pretraining mode
            self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=self.cfg.pretrain_learning_rate)

        loss_function_classification: Module = nn.CrossEntropyLoss()
        # actually the type is _Loss, but that's protected, for backward compatibility.
        # https://discuss.pytorch.org/t/why-is-the-pytorch-loss-base-class-protected/123417

        losses: List[Loss] = []
        scheduler = StepLR(self.optimizer, step_size=1, gamma=self.cfg.learning_rate_step)
        for current_epoch in range(n_epochs):
            print(f"[epoch {current_epoch}]")
            for n_current_batch, (inputs, labels) in enumerate(train_loader):
                loss, classification_loss = self._process_batch(loss_function_classification, inputs, labels,
                                                                critic_loader=critic_loader)
                losses.append(loss)

                if not self.cfg.logging_disabled:
                    if n_current_batch % log_interval == 0:
                        self.log_values(losses[-1], classification_loss, n_current_batch,
                                        learning_rate=self.optimizer.param_groups[0]['lr'])
                    if n_current_batch % self.cfg.log_interval_accuracy == 0 and test_loader:
                        self.log_accuracy(train_loader, test_loader, n_current_batch)
                        # global_step = self.global_step(n_current_batch)
                        ImageHandler.add_gradient_images(self.test_batch_for_visualization, self, "2: during training",
                                                         global_step=global_vars.global_step)
                    if not critic_loader:  # in pretraining mode
                        n_total_batch = current_epoch * self.cfg.n_training_batches + n_current_batch
                        progress_percentage: float = 100 * n_total_batch / self.cfg.pretraining_iterations
                        print(f'[pretraining iteration {n_total_batch} of {self.cfg.pretraining_iterations} '
                              f'({colored(200, 200, 100, f"{progress_percentage:.0f}%")})]')
                        global_vars.global_step += 1
            # self.update_epoch_writer_step_offset(train_loader)

            self.save_state(f"models/{utils.config_string(self.cfg)}.pt", epoch=n_epochs, loss=losses[-1])
            if not self.cfg.constant_lr:
                scheduler.step()
        self.terminate_writer()

        return losses[0], super()._smooth_end_losses(losses)

    def _process_batch(self, loss_function: nn.Module, inputs: Tensor, labels: Tensor,
                       critic_loader: DataLoader[Any] = None) -> Tuple[Loss, Loss]:

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.classifier(inputs)
        loss_classification = loss_function(outputs, labels)
        loss = self._add_explanation_loss(critic_loader, loss_classification)
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_classification.item()

    def log_values(self, loss, loss_classification, n_current_batch, learning_rate):

        self.add_scalars_to_writer(loss, loss_classification, learning_rate)
        self.print_statistics(loss, loss_classification, n_current_batch)

    def _add_explanation_loss(self, critic_loader, loss_classification):
        if critic_loader:
            loss_explanation = self.explanation_loss(critic_loader)
            loss = loss_classification + self.cfg.explanation_loss_weight * loss_explanation
        else:
            loss = loss_classification
        return loss

    def terminate_writer(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def print_statistics(self, loss, loss_classification, n_current_batch):
        # print statistics
        print(f'{colored(0,150,100, str(self.cfg.run_name))}: explainer [batch  {n_current_batch}] \n'
              f'Loss: {loss:.3f} = {loss_classification:.3f}(classification)'
              f' + {loss - loss_classification:.3f}(explanation)')
        if self.cfg.rtpt_enabled:
            self.rtpt.step(subtitle=f"loss={loss:2.2f}")

    # def global_step(self, n_current_batch: int) -> int:
    #     relative_step = n_current_batch * self.cfg.n_critic_batches if \
    #         self.cfg.n_critic_batches != 0 else n_current_batch
    #     return self.writer_step_offset + relative_step

    def add_scalars_to_writer(self, loss, loss_classification, learning_rate):
        global_step = global_vars.global_step
        if self.writer:
            self.writer.add_scalar("Explainer_Training/Explanation", loss - loss_classification,
                                   global_step=global_step)
            self.writer.add_scalar("Explainer_Training/Classification", loss_classification,
                                   global_step=global_step)
            self.writer.add_scalar("Explainer_Training/Total", loss, global_step=global_step)
            self.writer.add_scalar("Explainer_Training/Learning_Rate", learning_rate, global_step=global_step)

    def explanation_loss(self, critic_loader: DataLoader) -> float:
        critic = Critic(self.cfg, self.device, self.writer)
        explanations = []
        for inputs, labels in critic_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations.append(self.rescaled_input_gradient(inputs, labels))

        critic_end_of_training_loss: float
        # n_current_batch_total = (self.writer_step_offset // len(critic_loader)) + n_current_batch
        _, critic_end_of_training_loss = critic.train(critic_loader, explanations)

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
        return ImageHandler.rescale_to_zero_one(gradient)

    def predict(self, images: Tensor) -> Tensor:
        outputs = self.classifier(images)
        _, prediction = torch.max(outputs, 1)
        return prediction
