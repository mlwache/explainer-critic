import argparse
import warnings
# import sys
from typing import Iterator, Any, Tuple

from torch import Tensor
# from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

import torchvision
import torchvision.transforms as transforms

import torch.utils.data

from config import default_config as cfg
from explainer import Explainer
from visualization import Visualizer as Vis
# from rtpt import RTPT
# from net import Net

# Things necessary for Tensorboard Logging
import json
import os
from time import time
from torch.utils.tensorboard import SummaryWriter


def main():

    # creating rtpt object to name the process
    if cfg.rtpt_enabled:
        cfg.RTPT_OBJECT.start()

        # Tensorboard Writer
    log_dir = f"./runs/{int(time())}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    set_config_from_arguments(log_dir)

    print('Loading Data...')
    train_loader, test_loader, critic_loader = load_data()
    some_train_images, some_train_labels = get_one_batch_of_images(train_loader)
    some_test_images, some_test_labels = get_one_batch_of_images(test_loader)

    if cfg.render_enabled:
        print('Here are some training images and test images!')

        Vis.show_some_sample_images(some_train_images, some_train_labels)
        Vis.show_some_sample_images(some_test_images, some_test_labels)

    explainer = Explainer()

    if cfg.render_enabled:
        print('This is what the gradient looks like before training!')
        input_gradient: Tensor = explainer.input_gradient(some_test_images, some_test_labels)
        Vis.amplify_and_show(input_gradient)

    print(f'Training the Explainer on {cfg.n_training_samples} samples...')
    explainer.train(train_loader, critic_loader, writer)
    print('Finished Explainer Training')

    print(f'Saving the model to {cfg.PATH_TO_MODELS}.')
    explainer.save_model()
    print('Model Saved.')

    print('some images and their predictions:')
    print(f'Ground truth: {some_test_labels}')
    predicted = explainer.predict(some_test_images)
    print(f'predicted: {predicted}')

    print('Evaluating the Explainer')
    explainer_accuracy = explainer.compute_accuracy(test_loader)
    print(f'Explainer Accuracy on {cfg.n_test_samples} test images: {100 * explainer_accuracy} %')

    if cfg.render_enabled:
        print('This is what the gradient looks like after training!')
        input_gradient: Tensor = explainer.input_gradient(some_test_images, some_test_labels)
        Vis.amplify_and_show(input_gradient)

#         print('Showing the explainer\'s computation graph')
#         show_computation_graph(explainer.classifier, some_test_images)
#         print('Showing the critic\'s computation graph')
#         show_computation_graph(explainer.critic.classifier, some_test_images)
#
#
# def show_computation_graph(net: Net, images):
#     parameters = net.parameters()
#     y = net(images)
#     Vis.show_computation_graph(y, parameters)


# noinspection PyShadowingNames
def load_data() -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    mean_mnist = 0.1307
    std_dev_mnist = 0.3081
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(),
         torchvision.transforms.Normalize((mean_mnist,), (std_dev_mnist,))
         ])
    # transformation that first makes data to a tensor, and then normalizes them.
    # I took the mean and stddev from here:
    # https://nextjournal.com/gkoehler/pytorch-mnist (taking them as given for now)
    # maybe TODO: compute them myself, that seems more robust than taking magic numbers from the internet.

    with warnings.catch_warnings():  # Ignore warning, as it's caused by the underlying functional,
        # and I think would require me to change the site-packages in order to fix it.
        warnings.simplefilter("ignore")
        training_set: MNIST = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform_mnist)
        # loads the data to .data folder
        # ignores the UserWarning: The given NumPy array is not writeable,
        # and PyTorch does not support non-writeable tensors.
        # This means you can write to the underlying (supposedly non-writeable)
        # NumPy array using the tensor. You may want to copy the array to protect its data
        # or make it writeable before converting it to a tensor.
        # This type of warning will be suppressed for the rest of this program.

    # print(f"training set: {training_set}")
    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(training_set, batch_size=cfg.batch_size,
                                                                shuffle=True, num_workers=0)
    test_set: MNIST = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    critic_set: MNIST

    split = [int(len(test_set) * 0.5), int(len(test_set) * 0.5)]
    test_set, critic_set = torch.utils.data.random_split(test_set, split)

    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                                               shuffle=True, num_workers=0)
    critic_loader: DataLoader[Any] = torch.utils.data.DataLoader(critic_set, batch_size=cfg.batch_size,
                                                                 shuffle=False, num_workers=0)
    return train_loader, test_loader, critic_loader


def get_one_batch_of_images(loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
    data_iterator: Iterator[Any] = iter(loader)
    images: Tensor
    labels: Tensor
    images, labels = data_iterator.next()
    # The warning here is probably a PyCharm issue ([source](https://youtrack.jetbrains.com/issue/PY-12017))
    # I let Pycharm ignore the unresolved reference warning here.
    images = images.to(cfg.DEVICE)
    labels = labels.to(cfg.DEVICE)
    return images, labels


def set_config_from_arguments(log_dir: str):
    parser = argparse.ArgumentParser(description='Run the explainer-critic on MNIST.')
    config_dict = cfg.__dict__

    # Add an argument for all attributes in the config dataclass object
    for attribute_name in config_dict.keys():
        parser.add_argument(f'--{attribute_name}', type=type(config_dict[attribute_name]),
                            default=config_dict[attribute_name])
    passed_arguments = parser.parse_args()
    for attribute_name in config_dict.keys():
        passed_argument = getattr(passed_arguments, attribute_name)
        setattr(cfg, attribute_name, passed_argument)
        # globals()["cfg."+attribute_name] = argument
        # the left hand side here is the variable whose name is "cfg." + the string saved in attribute_name
    # Write config to log file
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json_dump: str = json.dumps(cfg.__dict__, default=lambda o: '<not serializable>')
        f.write(json_dump)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("No GPU found, falling back to CPU.")
        cfg.DEVICE = "cpu"

    # The following prevents there being too many open files at dl1.
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
