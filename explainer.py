from captum.attr import InputXGradient
import torch
import torch.nn as nn
# import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor
from typing import Any
from torch.optim import Optimizer

from torch.nn.modules import Module

from config import default_config as cfg
from critic import Critic

from net import Net
from rtpt import RTPT


class Explainer:
    classifier: nn.Module

    def __init__(self):
        self.classifier = Net(accepts_additional_explanations=False)
        self.critic = Critic()
        # print(net)

    def evaluate(self, test_loader: DataLoader[Any]):
        correct: int = 0
        total: int = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                if i >= cfg.n_test_batches:
                    break
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.classifier(images)
                # the class with the highest output is what we choose as prediction
                predicted: Tensor
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        total_accuracy = correct / total
        print(f'Accuracy of the explainer on the {total} test images: {100 * total_accuracy} %')
        return total_accuracy

    def save_model(self):
        torch.save(self.classifier.state_dict(), cfg.path_to_models)

    def train(self, train_loader: DataLoader[Any], critic_loader: DataLoader[Any], rtpt: RTPT):
        classification_loss: Module = cfg.loss  # actually the type should be _Loss.
        # TODO: (nice to have):
        # https://stackoverflow.com/questions/42736044/python-access-to-a-protected-member-of-a-class
        optimizer: Optimizer = cfg.optimizer(self.classifier.parameters())

        for epoch in range(cfg.n_epochs):

            # running_loss = 0.0
            for i, data in enumerate(train_loader, 0):  # i is the index of the current batch.

                # only train on a part of the samples.
                if i >= cfg.n_train_batches:
                    break

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.classifier(inputs)
                loss_classification = classification_loss(outputs, labels)
                loss_explanation = self.explanation_loss(critic_loader)
                loss = loss_classification + loss_explanation
                loss.backward()
                optimizer.step()

                # print statistics
                # running_loss += loss.item()
                if (i + 1) % (cfg.n_train_batches / 10) == 0:
                    # running_loss_average = running_loss / (cfg.n_train_batches / 10)
                    print(f'explainer[epoch {epoch}, batch {i+1}] \n'
                          f'Loss: {loss:.3f} = {loss_classification:.3f}(classification)'
                          f' + {loss_explanation:.3f}(explanation)')
                    # running_loss = 0.0
                if rtpt is not None:
                    rtpt.step(subtitle=f"loss={loss:2.2f}")

    def explanation_loss(self, critic_loader):
        explanations = []
        for _, data in enumerate(critic_loader, 0):
            inputs, labels = data
            explanations.append(self.input_gradient(inputs, labels))
        critic_end_of_training_loss: float = self.critic.train(critic_loader, explanations)
        return critic_end_of_training_loss

    def input_gradient(self, input_images: Tensor, labels: Tensor) -> Tensor:
        assert input_images.size() == torch.Size([cfg.batch_size, 1, 28, 28])
        assert labels.size() == torch.Size([cfg.batch_size])
        input_x_gradient = InputXGradient(self.classifier.forward)
        input_images.requires_grad = True
        gradient_x_input_one_image: Tensor = input_x_gradient.attribute(inputs=input_images, target=labels)
        gradient = gradient_x_input_one_image / input_images
        return gradient


    def print_prediction_one_batch(self, images, labels):
        print('GroundTruth: ', ' '.join('%5s' % cfg.classes[labels[j]] for j in range(cfg.batch_size)))
        # self.classifier.load_state_dict(torch.load(cfg.path_to_models))

        outputs = self.classifier(images)

        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % cfg.classes[predicted[j]]
                                      for j in range(cfg.batch_size)))
