import warnings

from captum.attr import InputXGradient
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from torch.utils.data import DataLoader
from torch import Tensor
from typing import Any

from torch.nn.modules import Module

from config import default_config as cfg


class Explainer:
    classifier: nn.Module

    def __init__(self):
        self.classifier = Net()
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

    def train(self, train_loader: DataLoader[Any]):
        criterion: Module = nn.CrossEntropyLoss()  # actually should be _Loss .
        # TODO: (nice to have):
        # https://stackoverflow.com/questions/42736044/python-access-to-a-protected-member-of-a-class
        optimizer = optim.SGD(self.classifier.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        # maybe TODO: outsource to config.

        for epoch in range(cfg.n_epochs):

            running_loss = 0.0
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
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i + 1) % (cfg.n_train_batches / 10) == 0:  # print the loss 10 times in total
                    print('[epoch %d, batch %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / (cfg.n_train_batches / 10)))
                    # (average over the last tenth of the batches)
                    running_loss = 0.0

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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # noinspection PyTypeChecker
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # I'm not sure why there's a warning here. It's still there when downloading their notebook,
        # so it might be a problem with the tutorial.
        # [here](https://stackoverflow.com/questions/48132786/why-is-this-warning-expected-type-int-matched-generic-type-t-got-dict)
        # it sounds like it's a PyCharm Issue.
        # ("My guess would be the analytics that give this warning are not sharp enough.")
        # I let PyCharm ignore it for now.

    def forward(self, x: Tensor) -> Tensor:
        assert x.size() == torch.Size([cfg.batch_size, 1, 28, 28])
        with warnings.catch_warnings():  # ignore the named tensor warning as it's not important,
            # and it adds visual clutter. (It's not important because I will keep the venv stable,
            # and my code is not critical for anyone.
            # "UserWarning: Named tensors and all their associated APIs are an experimental feature
            # and subject to change. Please do not use them for anything important
            # until they are released as stable."
            warnings.simplefilter("ignore")
            x = f.relu(f.max_pool2d(self.conv1(x), 2))
            x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = f.dropout(x, training=self.training)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=-1)
        assert x.size() == torch.Size([cfg.batch_size, len(cfg.classes)])
        return x   # Implicit dimension choice for log_softmax
        # has been deprecated. Just using the last dimension for now.
        # (https://stackoverflow.com/questions/49006773/userwarning-implicit-dimension-choice-for-log-softmax-has-been-deprecated)
