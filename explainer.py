import os
from typing import Any, Tuple, Optional

from captum.attr import IntegratedGradients

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
from utils import Logging
from utils import colored, Loaders
from visualization import ImageHandler

Loss = float


class Explainer:
    classifier: Net
    optimizer: Optimizer

    device: str
    loaders: Optional[Loaders]  # None if the explainer is only loaded from checkpoint, and not trained
    optimizer_type: Optional[str]
    logging: Optional[Logging]  # None if logging is disabled
    test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]]
    rtpt: Optional[RTPT]
    model_path: str

    def __init__(self, device: str, loaders: Optional[Loaders], optimizer_type: Optional[str],
                 logging: Optional[Logging], test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]],
                 rtpt: Optional[RTPT], model_path: str, explanation_mode: str):
        self.device = device
        self.loaders = loaders
        self.optimizer_type = optimizer_type
        self.logging = logging
        self.test_batch_to_visualize = test_batch_to_visualize
        self.rtpt = rtpt
        self.model_path = model_path
        self.explanation_mode = explanation_mode

        self.classifier = Net().to(device)

        # if explanation_mode == "integrated_gradient":
        #     self.classifier = self.classifier.double()
        if rtpt:
            self.rtpt.start()

    def load_state(self, path: str):
        if self.device == 'cuda':
            checkpoint: dict = torch.load(path)
        elif self.device == 'cpu':
            checkpoint: dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            raise NotImplementedError(f"dealing with device {self.device} not implemented")
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # currently not needed, as we don't use checkpoints to continue training, only for inference.
        self.classifier.train()

    def save_state(self, path: str, epoch: int, loss: float):
        if path and self.logging:  # empty model path means we don't save the model
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

    def pretrain(self,
                 learning_rate_start: float,
                 learning_rate_step: float,
                 constant_lr: bool,
                 n_epochs: int,
                 ) -> Tuple[Loss, Loss]:
        init_loss, end_loss = self.train(learning_rate_start, learning_rate_step, n_epochs, constant_lr,
                                         explanation_loss_weight=0.0, critic_lr=None)
        self.save_state('./models/pretrained_model.pt', epoch=-1, loss=end_loss)
        return init_loss, end_loss

    def train(self,
              learning_rate_start: float,
              learning_rate_step: float,
              n_epochs: int,
              constant_lr: bool,
              explanation_loss_weight: float,
              critic_lr: Optional[float]
              ) -> Tuple[Loss, Loss]:

        if self.loaders is None or self.optimizer_type is None:
            raise ValueError("Can't train, because the explainer is in evaluation mode.")

        self.classifier.train()
        self.initialize_optimizer(learning_rate_start)
        classification_loss_fn: Module = nn.CrossEntropyLoss()
        scheduler = StepLR(self.optimizer, step_size=1, gamma=learning_rate_step)

        start_classification_loss: Optional[Loss] = None
        end_classification_loss: Optional[Loss] = None
        mean_critic_loss: Loss = 0
        for current_epoch in range(n_epochs):
            print(f"epoch {current_epoch}")

            for n_current_batch, (inputs, labels) in enumerate(self.loaders.train):
                self.optimizer.zero_grad()

                outputs = self.classifier(inputs)
                classification_loss = classification_loss_fn(outputs, labels)

                explanation_loss_total_weight = 0.0
                if critic_lr is not None:  # if we are not in pretraining

                    # this will add to the gradients of the explainer classifier's weights
                    mean_critic_loss = self.train_critic_on_explanations(critic_lr)

                    # however, as the gradients of the critic loss are added in each critic step,
                    # they are divided by the length of the critic set so the length of the critic set does
                    # not influence the experiments by modulating the number of added gradients.
                    explanation_loss_total_weight = explanation_loss_weight / len(self.loaders.critic)
                    for x in self.classifier.parameters():
                        x.grad *= explanation_loss_total_weight

                # additionally, add the gradients of the classification loss
                classification_loss.backward()

                if n_current_batch == 0:
                    start_classification_loss = classification_loss.item()
                end_classification_loss = classification_loss.item()

                self.optimizer.step()
                self.log_values(classification_loss=classification_loss.item(),
                                pretraining_mode=critic_lr is None,
                                current_epoch=current_epoch,
                                n_current_batch=n_current_batch,
                                n_epochs=n_epochs,
                                mean_critic_loss=mean_critic_loss,
                                explanation_loss_total_weight=explanation_loss_total_weight)

            if self.device != 'cpu':  # on the cpu I assume it's not a valuable run which needs saving
                self.save_state(self.model_path, epoch=n_epochs, loss=end_classification_loss)
            if not constant_lr:
                scheduler.step()

        self.terminate_writer()
        return start_classification_loss, end_classification_loss

    def train_critic_on_explanations(self, critic_lr: float):

        critic = Critic(explanation_mode=self.explanation_mode,
                        device=self.device,
                        critic_loader=self.loaders.critic,
                        writer=self.logging.writer if self.logging else None,
                        log_interval_critic=self.logging.critic_log_interval if self.logging else None)
        explanations = []
        for inputs, labels in self.loaders.critic:
            if self.explanation_mode == "input_x_gradient" or self.explanation_mode == "gradient":
                explanations.append(self.rescaled_input_gradient(inputs, labels))
            elif self.explanation_mode == "integrated_gradient":
                explanations.append(ImageHandler.rescale_to_zero_one(self.integrated_gradient(inputs, labels)))
            else:
                raise NotImplementedError(f"unknown explanation mode '{self.explanation_mode}'")

        critic_mean_loss: float
        *_, critic_mean_loss = critic.train(explanations, critic_lr)

        return critic_mean_loss

    def log_values(self, classification_loss: float, pretraining_mode: bool, current_epoch: int,
                   n_current_batch: int, n_epochs: int, mean_critic_loss: float, explanation_loss_total_weight: float):
        if self.logging:
            if n_current_batch % self.logging.log_interval == 0:
                self.log_training_details(explanation_loss_total_weight=explanation_loss_total_weight,
                                          mean_critic_loss=mean_critic_loss,
                                          classification_loss=classification_loss,
                                          n_current_batch=n_current_batch,
                                          learning_rate=self.optimizer.param_groups[0]['lr'])
            if n_current_batch % self.logging.log_interval_accuracy == 0 and self.loaders.test:
                self.log_accuracy()
                ImageHandler.add_gradient_images(self.test_batch_to_visualize, self, "2: during training",
                                                 global_step=global_vars.global_step)
            if pretraining_mode:
                progress_percentage: float = 100 * current_epoch / n_epochs
                print(f'{colored(0, 150, 100, "pretraining:")} epoch {current_epoch}, '
                      f'batch {n_current_batch} of {n_epochs} epochs '
                      f'({colored(200, 200, 100, f"{progress_percentage:.0f}%")})]')
                # in pretraining mode the global step is not increased in the critic, so it needs to be done here.
                global_vars.global_step += 1

    def train_from_args(self, args: SimpleArgumentParser):
        return self.train(args.learning_rate_start, args.learning_rate_step, args.n_epochs,
                          args.constant_lr, args.explanation_loss_weight, args.learning_rate_critic)

    def pretrain_from_args(self, args: SimpleArgumentParser):
        return self.pretrain(args.pretrain_learning_rate, args.learning_rate_step, args.constant_lr,
                             args.n_pretraining_epochs)

    def log_training_details(self, explanation_loss_total_weight, mean_critic_loss, classification_loss,
                             n_current_batch, learning_rate):

        # add scalars to writer
        global_step = global_vars.global_step
        if self.logging:
            if explanation_loss_total_weight:
                total_loss = mean_critic_loss * explanation_loss_total_weight + classification_loss
            else:
                total_loss = classification_loss
            self.logging.writer.add_scalar("Explainer_Training/Explanation", mean_critic_loss,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Classification", classification_loss,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Total", total_loss,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Learning_Rate", learning_rate, global_step=global_step)

            # print statistics
            print(f'{colored(0, 150, 100, str(self.logging.run_name))}: explainer [batch  {n_current_batch}] \n'
                  f'Loss: {total_loss:.3f} ='
                  f' {classification_loss:.3f}(classification) + {explanation_loss_total_weight}(lambda)'
                  f'*{mean_critic_loss:.3f}(explanation)')
            if self.rtpt:
                self.rtpt.step(subtitle=f"loss={mean_critic_loss:2.2f}")

    def terminate_writer(self):
        if self.logging:
            self.logging.writer.flush()
            self.logging.writer.close()

    def integrated_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        integrated_gradients = IntegratedGradients(self.classifier.forward)
        input_images.requires_grad = True
        int_grad: Tensor = integrated_gradients.attribute(inputs=input_images, target=labels)
        int_grad = int_grad.float()
        return int_grad

    def input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        input_x_gradient = InputXGradient(self.classifier.forward)
        input_images.requires_grad = True
        gradient_x_input: Tensor = input_x_gradient.attribute(inputs=input_images, target=labels)
        gradient: Tensor = gradient_x_input / input_images
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
        test_accuracy = self.compute_accuracy(self.loaders.test, self.logging.n_test_batches)
        print(colored(0, 0, 200, f'accuracy training: {training_accuracy}, accuracy testing: {test_accuracy:.3f}'))
        if self.logging:
            self.logging.writer.add_scalar("Explainer_Training/Training_Accuracy", training_accuracy,
                                           global_step=global_step)
            self.logging.writer.add_scalar("Explainer_Training/Test_Accuracy", test_accuracy,
                                           global_step=global_step)

    def initialize_optimizer(self, learning_rate):
        if self.optimizer_type == "adadelta":
            self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"optimizer '{self.optimizer_type}' invalid")
