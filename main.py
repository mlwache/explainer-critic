import warnings
from typing import Iterator, Any

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import torchvision
import torchvision.transforms as transforms

import torch.utils.data

from config import default_config as cfg
from explainer import Explainer
from visualization import Visualizer as Vis


def main():
    print('Loading Data...')
    train_loader, test_loader, critic_loader = load_data()
    train_images, train_labels = get_some_random_images(train_loader)
    test_images, test_labels = get_some_random_images(test_loader)

    print('Here are some training images and test images!')

    Vis.show_some_sample_images(train_images, train_labels)
    Vis.show_some_sample_images(test_images, test_labels)

    explainer = Explainer()

    print('This is what the gradient looks like before training!')
    input_gradient: Tensor = explainer.input_gradient(test_images, test_labels)
    Vis.amplify_and_show(input_gradient)

    print(f'Training the Explainer on {cfg.n_training_samples} samples.')
    explainer.train(train_loader, critic_loader)
    print('Finished Explainer Training')

    print(f'Saving the model to {cfg.path_to_models}.')
    explainer.save_model()
    print('Model Saved.')

    print('Showing some images with their prediction')
    explainer.print_prediction_one_batch(test_images, test_labels)

    print('Evaluating the Explainer')
    explainer.evaluate(test_loader)

    print('This is what the gradient looks like after training!')
    input_gradient: Tensor = explainer.input_gradient(test_images, test_labels)
    Vis.amplify_and_show(input_gradient)


# noinspection PyShadowingNames
def load_data() -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    transform_mnist = transforms.Compose(
        [transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))
         ])
    # transformation that first makes data to a tensor, and then normalizes them.
    # the numbers for normalization are the global mean and standard deviation of MNIST
    # I took them from here: https://nextjournal.com/gkoehler/pytorch-mnist (taking them as given for now)
    # TODO: compute them myself, that seems more robust than taking magic numbers from the internet.

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
                                                                shuffle=True, num_workers=2)
    test_set: MNIST = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    split = [int(len(test_set) * 0.5), int(len(test_set) * 0.5)]
    critic_set = torch.utils.data.random_split(test_set, split)[0].dataset
    # critic_set: MNIST = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)

    # print(f"test set: {test_set}")
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                                               shuffle=True, num_workers=2)
    critic_loader: DataLoader[Any] = torch.utils.data.DataLoader(critic_set, batch_size=cfg.batch_size,
                                                                 shuffle=True, num_workers=2)
    return train_loader, test_loader, critic_loader


def get_some_random_images(loader: DataLoader[Any]) -> tuple[Tensor, Tensor]:
    data_iterator: Iterator[Any] = iter(loader)
    images: Tensor
    images, labels = data_iterator.next()
    # The warning here is probably a PyCharm issue ([source](https://youtrack.jetbrains.com/issue/PY-12017))
    # I let Pycharm ignore the unresolved reference warning here.
    return images, labels


if __name__ == '__main__':
    main()
