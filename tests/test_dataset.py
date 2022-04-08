import torch

import utils
from config import SimpleArgumentParser


def test_deterministic_dataset():
    args: SimpleArgumentParser
    args = utils.parse_args([])  # use the default arguments instead of the input
    utils.setup(args, eval_mode=True)
    loaders = utils.load_data_from_args(args)
    # assert args.n_test_batches == len(loaders.test)

    input_batch, labels = iter(loaders.train).__next__()

    assert all(labels == torch.Tensor([5, 6, 7, 8, 5, 6, 5, 8, 9, 4, 1, 0, 2, 1, 4, 8, 9, 3, 4, 6, 8, 6, 8, 9,
                                       9, 1, 9, 4, 2, 6, 7, 4, 6, 4, 4, 5, 4, 2, 2, 2, 6, 1, 3, 8, 9, 8, 1, 9,
                                       2, 1, 7, 3, 3, 6, 4, 1, 5, 5, 8, 9, 6, 7, 0, 9, 0, 9, 1, 4, 8, 8, 9, 7,
                                       0, 7, 7, 6, 4, 1, 1, 0, 4, 5, 4, 6, 9, 1, 6, 4, 1, 3, 5, 9, 2, 1, 6, 5,
                                       1, 7, 8, 9, 9, 2, 3, 3, 7, 8, 9, 9, 7, 0, 5, 2, 7, 1, 0, 0, 7, 5, 0, 8,
                                       2, 4, 6, 4, 6, 6, 9, 5]))

    first_image = input_batch[0][0]

    # upper left pixel should be normalized, and black.
    assert (first_image[0][0] - (-0.4242)) < 0.0001


def test_loaders_to_tensor():
    utils.set_device()
    loaders = utils.load_data(10,10,10,5,5)

    inputs, labels = utils.loader_to_tensors(loaders.test)
    assert inputs.size() == torch.Size([10,1,28,28])
    assert labels.size() == torch.Size([10])


if __name__ == '__main__':
    test_deterministic_dataset()
