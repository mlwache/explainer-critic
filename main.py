# imports necessary for Tensorboard Logging
import json
import os
import warnings
from typing import Iterator, Any, Tuple

import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# torch related imports
from rtpt import RTPT
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST

# local imports
from config import SimpleArgumentParser
from explainer import Explainer
from visualization import Visualizer as Vis


def main(args: SimpleArgumentParser):
    print('Loading Data...')
    train_loader, test_loader, critic_loader = load_data(args)
    some_train_images, some_train_labels = get_one_batch_of_images(train_loader, args)
    some_test_images, some_test_labels = get_one_batch_of_images(test_loader, args)

    if args.render_enabled:
        print('Here are some training images and test images!')

        Vis.show_some_sample_images(some_train_images, some_train_labels, args)
        Vis.show_some_sample_images(some_test_images, some_test_labels, args)
        # Todo: add to config (or to dataset?)
        mean_mnist = 0.1307
        std_dev_mnist = 0.3081
        some_train_images = some_train_images * std_dev_mnist + mean_mnist
        combined_image = torchvision.utils.make_grid(some_train_images)
        args.WRITER.add_image("some training images", combined_image)

    explainer = Explainer(args)

    if args.render_enabled:
        print('This is what the gradient looks like before training!')
        input_gradient: Tensor = explainer.input_gradient(some_test_images, some_test_labels)
        Vis.amplify_and_show(input_gradient, args)
        amplified_gradient = Vis.amplify(input_gradient, args)
        grid_grad = torchvision.utils.make_grid(amplified_gradient)
        args.WRITER.add_image("gradient", grid_grad)

    print(f'Training the Explainer on {args.n_training_samples} samples...')
    explainer.train(train_loader=train_loader, critic_loader=critic_loader)
    print('Finished Explainer Training')

    print(f'Saving the model to {args.PATH_TO_MODELS}.')
    explainer.save_model()
    print('Model Saved.')

    print('some images and their predictions:')
    print(f'Ground truth: {some_test_labels}')
    predicted = explainer.predict(some_test_images)
    print(f'predicted: {predicted}')

    print('Evaluating the Explainer')
    explainer_accuracy = explainer.compute_accuracy(test_loader)
    print(f'Explainer Accuracy on {args.n_test_samples} test images: {100 * explainer_accuracy} %')

    if args.render_enabled:
        print('This is what the gradient looks like after training!')
        input_gradient: Tensor = explainer.input_gradient(some_test_images, some_test_labels)
        Vis.amplify_and_show(input_gradient, args)


# noinspection PyShadowingNames
def load_data(cfg) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    # dataset splits for the different parts of training and testing
    training_set: MNIST
    critic_set: MNIST
    test_set: MNIST

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
        training_and_critic_set: MNIST = torchvision.datasets.MNIST('./data', train=True, download=True,
                                                                    transform=transform_mnist)
        # loads the data to .data folder
        # ignores the UserWarning: The given NumPy array is not writeable,
        # and PyTorch does not support non-writeable tensors.
        # This means you can write to the underlying (supposedly non-writeable)
        # NumPy array using the tensor. You may want to copy the array to protect its data
        # or make it writeable before converting it to a tensor.
        # This type of warning will be suppressed for the rest of this program.

    n_spare_samples = len(training_and_critic_set) - cfg.n_training_samples - cfg.n_critic_samples
    assert n_spare_samples >= 0
    split = [cfg.n_training_samples, cfg.n_critic_samples, n_spare_samples]
    training_set, critic_set, _ = torch.utils.data.random_split(training_and_critic_set, split)

    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(training_set, batch_size=cfg.batch_size,
                                                                shuffle=True, num_workers=0)
    critic_loader: DataLoader[Any] = torch.utils.data.DataLoader(critic_set, batch_size=cfg.batch_size,
                                                                 shuffle=False, num_workers=0)

    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform_mnist)
    split = [cfg.n_test_samples, len(test_set) - cfg.n_test_samples]
    test_set, _ = torch.utils.data.random_split(test_set, split)
    test_loader: DataLoader[Any] = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                                               shuffle=True, num_workers=0)
    return train_loader, test_loader, critic_loader


def get_one_batch_of_images(loader: DataLoader[Any], cfg) -> Tuple[Tensor, Tensor]:
    data_iterator: Iterator[Any] = iter(loader)
    images: Tensor
    labels: Tensor
    images, labels = data_iterator.next()
    # The warning here is probably a PyCharm issue ([source](https://youtrack.jetbrains.com/issue/PY-12017))
    # I let Pycharm ignore the unresolved reference warning here.
    images = images.to(cfg.DEVICE)
    labels = labels.to(cfg.DEVICE)
    return images, labels


def set_sharing_strategy():
    # The following prevents there being too many open files at dl1.
    torch.multiprocessing.set_sharing_strategy('file_system')


def start_rtpt(rtpt_object: RTPT):
    # using rtpt object to name the process
    if rtpt_object is not None:
        rtpt_object.start()


def write_config_to_log(arg: SimpleArgumentParser):
    # Write config to log file
    with open(os.path.join(arg.LOG_DIR, "config.json"), 'w') as f:
        json_dump: str = json.dumps(arg.__dict__, default=lambda o: '<not serializable>')
        f.write(json_dump)


def setup() -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    args.parse_args()
    set_sharing_strategy()
    write_config_to_log(args)
    start_rtpt(args.RTPT_OBJECT)
    return args


if __name__ == '__main__':
    arguments = setup()
    main(arguments)
