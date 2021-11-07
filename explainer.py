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

import global_vars
from config import SimpleArgumentParser
from critic import Critic
from net import Net
from utils import Logging, smooth_end_losses
from utils import colored, Loaders
from visualization import ImageHandler

Loss = float


class Explainer:
    classifier: Net
    optimizer: Optimizer
    test_batch_for_visualization: Optional[Tuple[Tensor, Tensor]]
    logging: Optional[Logging]  # None if logging is disabled
    loaders: Optional[Loaders]  # None if the explainer is only loaded from checkpoint, and not trained
    model_path: str
    device: str
    rtpt: Optional[RTPT]

    def __init__(self, device: str, loaders: Optional[Loaders], logging: Optional[Logging],
                 test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]], rtpt: Optional[RTPT],
                 model_path: str):
        self.device = device
        self.logging = logging
        self.loaders = loaders
        self.test_batch_for_visualization = test_batch_to_visualize
        self.rtpt = rtpt
        self.model_path = model_path

        self.classifier = Net().to(device)
        if rtpt:
            self.rtpt.start()

    def load_state(self, path: str):
        checkpoint: dict = torch.load(path)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # currently not needed, as we don't use checkpoints to continue training, only for inference.
        self.classifier.train()

    def save_state(self, path: str, epoch: int = -1, loss: float = -1.0):
        if path:  # empty model path means we don't save the model
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

    def pre_train(self,
                  learning_rate_start: float,
                  learning_rate_step: float,
                  constant_lr: bool,
                  n_epochs: int,
                  ) -> Tuple[Loss, Loss]:
        init_loss, end_loss = self.train(learning_rate_start, learning_rate_step, n_epochs, constant_lr,
                                         explanation_loss_weight=0.0, critic_lr=0.0)
        self.save_state('./models/pretrained_model.pt', epoch=-1, loss=end_loss)
        return init_loss, end_loss

    def train_from_args(self, args: SimpleArgumentParser):
        return self.train(args.learning_rate_start, args.learning_rate_step, args.n_epochs,
                          args.constant_lr, args.explanation_loss_weight, args.learning_rate_critic)

    def train(self,
              learning_rate_start: float,
              learning_rate_step: float,
              n_epochs: int,
              constant_lr: bool,
              explanation_loss_weight: float,
              critic_lr: float
              ) -> Tuple[Loss, Loss]:
        self.classifier.train()

        self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=learning_rate_start)

        loss_function_classification: Module = nn.CrossEntropyLoss()

        losses: List[Loss] = []
        scheduler = StepLR(self.optimizer, step_size=1, gamma=learning_rate_step)
        for current_epoch in range(n_epochs):
            print(f"[epoch {current_epoch}]")
            for n_current_batch, (inputs, labels) in enumerate(self.loaders.train):
                loss, classification_loss = self._process_batch(loss_function_classification, inputs,
                                                                labels, explanation_loss_weight, critic_lr)
                losses.append(loss)

                if self.logging:
                    if n_current_batch % self.logging.log_interval == 0:
                        self.log_values(losses[-1], classification_loss, n_current_batch,
                                        learning_rate=self.optimizer.param_groups[0]['lr'])
                    if n_current_batch % self.logging.log_interval_accuracy == 0 and self.loaders.test:
                        self.log_accuracy()
                        # global_step = self.global_step(n_current_batch)
                        ImageHandler.add_gradient_images(self.test_batch_for_visualization, self, "2: during training",
                                                         global_step=global_vars.global_step)
                    if not critic_lr:  # in pretraining mode
                        progress_percentage: float = 100 * current_epoch / n_epochs
                        print(f'{colored(0, 150, 100, "pretraining:")} epoch {current_epoch}, '
                              f'batch {n_current_batch} of {n_epochs} epochs '
                              f'({colored(200, 200, 100, f"{progress_percentage:.0f}%")})]')
                        global_vars.global_step += 1

            self.save_state(self.model_path, epoch=n_epochs, loss=losses[-1])
            if not constant_lr:
                scheduler.step()
        self.terminate_writer()

        return losses[0], smooth_end_losses(losses)

    def _process_batch(self, loss_function: nn.Module, inputs: Tensor, labels: Tensor,
                       explanation_loss_weight, critic_lr: float) -> Tuple[Loss, Loss]:

        # inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.classifier(inputs)
        loss_classification = loss_function(outputs, labels)
        loss = self._add_explanation_loss(loss_classification, explanation_loss_weight, critic_lr)
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_classification.item()

    def log_values(self, loss, loss_classification, n_current_batch, learning_rate):

        self.add_scalars_to_writer(loss, loss_classification, learning_rate)
        self.print_statistics(loss, loss_classification, n_current_batch)

    def _add_explanation_loss(self, loss_classification, explanation_loss_weight, critic_lr):
        if critic_lr:
            loss_explanation = self.explanation_loss(self.loaders.critic, critic_lr)
            loss = loss_classification + explanation_loss_weight * loss_explanation
        else:
            loss = loss_classification
        return loss

    def terminate_writer(self):
        if self.logging:
            self.logging.writer.flush()
            self.logging.writer.close()

    def print_statistics(self, loss, loss_classification, n_current_batch):
        # print statistics
        print(f'{colored(0, 150, 100, str(self.logging.run_name))}: explainer [batch  {n_current_batch}] \n'
              f'Loss: {loss:.3f} = {loss_classification:.3f}(classification)'
              f' + {loss - loss_classification:.3f}(explanation)')
        if self.rtpt:
            self.rtpt.step(subtitle=f"loss={loss:2.2f}")

    def add_scalars_to_writer(self, loss, loss_classification, learning_rate):
        global_step = global_vars.global_step
        if self.logging:
            self.logging.writer.add_scalar("Explainer_Training/Explanation", loss - loss_classification,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Classification", loss_classification,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Total", loss, global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Learning_Rate", learning_rate, global_step=global_step)

    def explanation_loss(self, critic_loader: DataLoader, critic_lr: float) -> float:
        critic = Critic(self.device, critic_loader, self.logging.writer if self.logging else None,
                        self.logging.critic_log_interval if self.logging else -1)
        explanations = []
        for inputs, labels in critic_loader:
            # inputs, labels = inputs.to(self.device), labels.to(self.device)
            explanations.append(self.rescaled_input_gradient(inputs, labels))

        critic_end_of_training_loss: float
        _, critic_end_of_training_loss = critic.train(explanations, critic_lr)

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

    def compute_accuracy(self, data_loader: DataLoader[Any], n_batches: int):
        n_correct_samples: int = 0
        n_test_samples_total: int = 0

        self.classifier.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if i >= n_batches:  # only test on a set of the test set size, even for training accuracy.
                    break
                # images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.classifier(images)

                # the class with the highest output is what we choose as prediction
                _, predicted = torch.max(outputs.data, dim=1)
                n_test_samples_total += labels.size()[0]
                n_correct_samples += (predicted == labels).sum().item()
        total_accuracy = n_correct_samples / n_test_samples_total
        self.classifier.train()
        return total_accuracy

    def log_accuracy(self):
        global_step = global_vars.global_step
        training_accuracy = self.compute_accuracy(self.loaders.train, self.logging.n_test_batches)
        # test_accuracy = -0.1  # just for initializing. negative so that we will notice if it's unchanged
        test_accuracy = self.compute_accuracy(self.loaders.test, self.logging.n_test_batches)
        print(colored(0, 0, 200, f'accuracy training: {training_accuracy}, accuracy testing: {test_accuracy:.3f}'))
        if self.logging:
            self.logging.writer.add_scalar("Explainer_Training/Training_Accuracy", training_accuracy,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Test_Accuracy", test_accuracy,
                                           global_step=global_step)
