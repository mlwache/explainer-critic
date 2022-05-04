import os
from typing import Tuple, Optional, List

import torch
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from torch import Tensor, nn, optim
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import global_vars
import utils
from config import SimpleArgumentParser
from critic import Critic
from net import Net
from utils import compute_accuracy
from utils import colored, Loaders
from visualization import ImageHandler

Loss = float


class Explainer:
    classifier: Net
    optimizer: Optimizer

    loaders: Optional[Loaders]  # None if the explainer is only loaded from checkpoint, and not trained
    optimizer_type: Optional[str]
    test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]]
    model_path: str

    def __init__(self,
                 loaders: Optional[Loaders],
                 optimizer_type: Optional[str],
                 test_batch_to_visualize: Optional[Tuple[Tensor, Tensor]],
                 model_path: str, explanation_mode: str):
        self.loaders = loaders
        self.optimizer_type = optimizer_type
        self.test_batch_to_visualize = test_batch_to_visualize
        self.model_path = model_path
        self.explanation_mode = explanation_mode
        self.critic: Optional[Critic] = None

        self.classifier = Net().to(global_vars.DEVICE)

    def load_state(self, path: str):
        path = os.path.join(utils.get_git_root(), path)
        if global_vars.DEVICE == 'cuda':
            checkpoint: dict = torch.load(path)
        elif global_vars.DEVICE == 'cpu':
            checkpoint: dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            raise NotImplementedError(f"dealing with device {global_vars.DEVICE} not implemented")
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # currently not needed, as we don't use checkpoints to continue training, only for inference.
        self.classifier.train()

    def save_state(self, path: str, epoch: int, loss: float):
        if path and global_vars.LOGGING:  # empty model path means we don't save the model
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
                 learning_rate: float,
                 learning_rate_step: float,
                 lr_scheduling: bool,
                 n_epochs: int,
                 ) -> Tuple[Loss, Loss]:
        init_loss, end_loss = self.train(learning_rate=learning_rate,
                                         learning_rate_step=learning_rate_step,
                                         n_epochs=n_epochs,
                                         lr_scheduling=lr_scheduling,
                                         explanation_loss_weight=0.0,
                                         critic_lr=None)
        self.save_state('./models/pretrained_model.pt', epoch=-1, loss=end_loss)
        return init_loss, end_loss

    def train(self,
              learning_rate: float,
              learning_rate_step: float,
              n_epochs: int,
              lr_scheduling: bool,
              explanation_loss_weight: float,
              critic_lr: Optional[float],
              shuffle_critic: bool = True
              ) -> Tuple[Loss, Loss]:

        if self.loaders is None or self.optimizer_type is None:
            raise ValueError("Can't train, because the explainer is in evaluation mode.")

        self.classifier.train()
        self.initialize_optimizer(learning_rate)
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
                    mean_critic_loss = self.train_critic_on_explanations(critic_lr=critic_lr,
                                                                         shuffle_critic=shuffle_critic)

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

            if global_vars.DEVICE != 'cpu':  # on the cpu I assume it's not a valuable run which needs saving
                self.save_state(self.model_path, epoch=n_epochs, loss=end_classification_loss)
            if lr_scheduling:
                scheduler.step()

        self.terminate_writer()
        return start_classification_loss, end_classification_loss

    def train_critic_on_explanations(self,
                                     critic_lr: float,
                                     shuffle_critic: bool,
                                     explanation_mode: Optional[str] = None):
        global_vars.global_step = 0 # delete this again.
        if explanation_mode is None:
            explanation_mode = self.explanation_mode
        self.critic = Critic(explanation_mode=explanation_mode,
                             critic_loader=self.loaders.critic,
                             log_interval_critic=global_vars.LOGGING.critic_log_interval if
                             global_vars.LOGGING else None,
                             shuffle_data=shuffle_critic)
        explanation_batches = [x for [x, _] in self.get_labeled_explanation_batches(self.loaders.critic,
                                                                                    explanation_mode)]
        critic_mean_loss: float
        *_, critic_mean_loss = self.critic.train(explanation_batches, critic_lr)

        return critic_mean_loss

    def get_labeled_explanation_batches(self,
                                        dataloader: DataLoader,
                                        explanation_mode: Optional[str] = None) -> List[List[Tensor]]:
        labeled_explanation_batches = []
        for inputs, labels in dataloader:
            labeled_explanation_batches.append([self.get_explanation_batch(inputs, labels, explanation_mode), labels])
        return labeled_explanation_batches

    def log_values(self, classification_loss: float, pretraining_mode: bool, current_epoch: int,
                   n_current_batch: int, n_epochs: int, mean_critic_loss: float, explanation_loss_total_weight: float):
        if global_vars.LOGGING:
            if n_current_batch % global_vars.LOGGING.log_interval == 0:
                self.log_training_details(explanation_loss_total_weight=explanation_loss_total_weight,
                                          mean_critic_loss=mean_critic_loss,
                                          classification_loss=classification_loss,
                                          learning_rate=self.optimizer.param_groups[0]['lr'])
            if n_current_batch % global_vars.LOGGING.log_interval_accuracy == 0 and self.loaders.test:
                self.log_accuracy()
                ImageHandler.add_gradient_images(self.test_batch_to_visualize, self, "2: during training",
                                                 global_step=global_vars.global_step)
            if pretraining_mode:
                print(f'{colored(100, 50, 100, "pretraining:")}')
                global_vars.global_step += 1
                # in pretraining mode the global step is not increased in the critic, so it needs to be done here.

            progress_percentage: float = 100 * current_epoch / n_epochs
            print(f'{colored(0, 150, 100, str(global_vars.LOGGING.run_name))}: '
                  f'epoch {current_epoch}, '
                  f'explainer batch {n_current_batch} of {n_epochs} epochs '
                  f'({colored(200, 200, 100, f"{progress_percentage:.0f}%")})]')

    def train_from_args(self, args: SimpleArgumentParser):
        return self.train(learning_rate=args.learning_rate,
                          learning_rate_step=args.learning_rate_step,
                          n_epochs=args.n_epochs,
                          lr_scheduling=args.lr_scheduling,
                          explanation_loss_weight=args.explanation_loss_weight,
                          critic_lr=args.learning_rate_critic,
                          shuffle_critic=not args.disable_critic_shuffling)

    def pretrain_from_args(self, args: SimpleArgumentParser):
        return self.pretrain(args.pretrain_learning_rate, args.learning_rate_step, args.lr_scheduling,
                             args.n_pretraining_epochs)

    @staticmethod
    def log_training_details(explanation_loss_total_weight, mean_critic_loss, classification_loss,
                             learning_rate):

        # add scalars to writer
        global_step = global_vars.global_step
        if global_vars.LOGGING:
            if explanation_loss_total_weight:
                total_loss = mean_critic_loss * explanation_loss_total_weight + classification_loss
            else:
                total_loss = classification_loss
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Explanation", mean_critic_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Classification", classification_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Total", total_loss,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Learning_Rate", learning_rate,
                                                  global_step=global_step)

            # print statistics
            print(f'Loss: {total_loss:.3f} ='
                  f' {classification_loss:.3f}(classification) + {explanation_loss_total_weight}(lambda)'
                  f'*{mean_critic_loss:.3f}(explanation)')

    @staticmethod
    def terminate_writer():
        if global_vars.LOGGING:
            global_vars.LOGGING.writer.flush()
            global_vars.LOGGING.writer.close()

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

    @staticmethod
    def clip_and_rescale(images: Tensor) -> Tensor:
        # if not self.disable_gradient_clipping:
        # clip negative gradients to zero (don't distinguish between "no impact" and "negative impact" on the label)
        images[images < 0] = 0
        return ImageHandler.rescale_to_zero_one(images)

    def get_explanation_batch(self, inputs: Tensor, labels: Tensor, explanation_mode: Optional[str] = None) -> Tensor:
        if explanation_mode is None:
            explanation_mode = self.explanation_mode

        if explanation_mode == "gradient" or explanation_mode == "input_x_gradient":
            input_gradient = self.input_gradient(inputs, labels)
            clipped_rescaled_input_gradient = self.clip_and_rescale(input_gradient)
            if explanation_mode == "input_x_gradient":
                return clipped_rescaled_input_gradient * inputs
            else:
                return clipped_rescaled_input_gradient
        elif explanation_mode == "integrated_gradient" or explanation_mode == "input_x_integrated_gradient":
            integrated_gradient = self.integrated_gradient(inputs, labels)
            clipped_rescaled_integrated_gradient = self.clip_and_rescale(integrated_gradient)
            if self.explanation_mode == "input_x_integrated_gradient":
                return clipped_rescaled_integrated_gradient * inputs
            else:
                return clipped_rescaled_integrated_gradient
        elif explanation_mode == "input":
            return inputs
        else:
            raise NotImplementedError(f"unknown explanation mode '{explanation_mode}'")

    def predict(self, images: Tensor) -> Tensor:
        outputs = self.classifier(images)
        _, prediction = torch.max(outputs, 1)
        return prediction

    def log_accuracy(self):
        global_step = global_vars.global_step
        training_accuracy = compute_accuracy(self.classifier, self.loaders.train, global_vars.LOGGING.n_test_batches)
        test_accuracy = compute_accuracy(self.classifier, self.loaders.test)
        if self.critic:
            critic_test_accuracy = compute_accuracy(classifier=self.critic.classifier,
                                                    data=self.get_labeled_explanation_batches(self.loaders.test)
                                                    )
            critic_training_accuracy = compute_accuracy(classifier=self.critic.classifier,
                                                        data=self.get_labeled_explanation_batches(self.loaders.critic),
                                                        n_batches=len(self.loaders.test)
                                                        )
            critic_test_accuracy_input = compute_accuracy(classifier=self.critic.classifier,
                                                          data=self.loaders.test,
                                                          )
            critic_training_accuracy_input = compute_accuracy(classifier=self.critic.classifier,
                                                              data=self.loaders.critic,
                                                              n_batches=len(self.loaders.test)
                                                              )
        else:
            critic_test_accuracy = 0
            critic_training_accuracy = 0
            critic_test_accuracy_input = 0
            critic_training_accuracy_input = 0

        print(colored(0, 0, 200, f'accuracy training: {training_accuracy:3f}, accuracy testing: {test_accuracy:.3f}, '
                                 f'accuracy critic training:{critic_training_accuracy:3f}, accuracy critic testing:'
                                 f'{critic_test_accuracy:3f}'))
        if global_vars.LOGGING:
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Training_Accuracy", training_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Explainer_Training/Test_Accuracy", test_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Training_Accuracy", critic_training_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Test_Accuracy", critic_test_accuracy,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Input_Test_Accuracy", critic_test_accuracy_input,
                                                  global_step=global_step)
            global_vars.LOGGING.writer.add_scalar("Critic_Training/Input_Training_Accuracy",
                                                  critic_training_accuracy_input,
                                                  global_step=global_step)

    def initialize_optimizer(self, learning_rate):
        if self.optimizer_type == "adadelta":
            self.optimizer = optim.Adadelta(self.classifier.parameters(), lr=learning_rate)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"optimizer '{self.optimizer_type}' invalid")

    def get_labeled_explanations(self, test_loader: DataLoader, mode: str) -> Tuple[Tensor, Tensor]:
        """get all explanations together with the labels, and don't combine them into batches."""
        explanations = []
        explanation_labels = []
        for inputs, labels in test_loader:
            explanation_batch: List[Tensor] = list(self.get_explanation_batch(inputs, labels, mode))
            # labeled_explanation_batch: List[Tuple[Tensor, int]] = list(zip(explanation_batch, list(labels)))
            explanations.extend(explanation_batch)
            explanation_labels.extend(labels)
        explanation_tensor = torch.stack(explanations)
        label_tensor = torch.stack(explanation_labels)
        return explanation_tensor, label_tensor
