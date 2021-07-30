import warnings
import sys
from typing import Iterator, Any, Tuple

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import torchvision
import torchvision.transforms as transforms

import torch.utils.data

from config import default_config as cfg
from explainer import Explainer
from visualization import Visualizer as Vis
from rtpt import RTPT
# from net import Net


def main(render=False, use_rtpt=False):

    # creating rtpt object to name the process
    rtpt = None
    if use_rtpt:
        rtpt = RTPT(name_initials='MW', experiment_name='Explainer-Critic', max_iterations=cfg.n_train_batches)
        rtpt.start()

    print('Loading Data...')
    train_loader, test_loader, critic_loader = load_data()
    some_train_images, some_train_labels = get_one_batch_of_images(train_loader)
    some_test_images, some_test_labels = get_one_batch_of_images(test_loader)

    if render:
        print('Here are some training images and test images!')

        Vis.show_some_sample_images(some_train_images, some_train_labels)
        Vis.show_some_sample_images(some_test_images, some_test_labels)

    explainer = Explainer()

    if render:
        print('This is what the gradient looks like before training!')
        input_gradient: Tensor = explainer.input_gradient(some_test_images, some_test_labels)
        Vis.amplify_and_show(input_gradient)

    print(f'Training the Explainer on {cfg.n_training_samples} samples...')
    explainer.train(train_loader, critic_loader, rtpt)
    print('Finished Explainer Training')

    print(f'Saving the model to {cfg.path_to_models}.')
    explainer.save_model()
    print('Model Saved.')

    print('some images and their predictions:')
    print(f'Ground truth: {some_test_labels}')
    predicted = explainer.predict(some_test_images)
    print(f'predicted: {predicted}')

    print('Evaluating the Explainer')
    explainer_accuracy = explainer.compute_accuracy(test_loader)
    print(f'Explainer Accuracy on {cfg.n_test_samples} test images: {100 * explainer_accuracy} %')

    if render:
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


def get_one_batch_of_images(loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
    data_iterator: Iterator[Any] = iter(loader)
    images: Tensor
    labels: Tensor
    images, labels = data_iterator.next()
    # The warning here is probably a PyCharm issue ([source](https://youtrack.jetbrains.com/issue/PY-12017))
    # I let Pycharm ignore the unresolved reference warning here.
    return images, labels


if __name__ == '__main__':

    render_enabled = (sys.argv.__contains__("--render"))
    rtpt_enabled = (sys.argv.__contains__("--rtpt"))
    main(render=render_enabled, use_rtpt=rtpt_enabled)
