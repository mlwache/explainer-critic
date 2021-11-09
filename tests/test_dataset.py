import torch

import utils
from config import SimpleArgumentParser


def test_deterministic_dataset():
    args: SimpleArgumentParser
    args, *_ = utils.setup([], eval_mode=True)
    loaders = utils.load_data_from_args(args)
    # assert args.n_test_batches == len(loaders.test)

    input_batch, labels = iter(loaders.train).__next__()

    assert all(labels == torch.Tensor([6, 7, 6, 9, 4, 2, 1, 3, 8, 9, 7, 0, 3, 1, 2, 4, 6, 1, 4, 5, 3, 6, 5, 5,
                                       6, 5, 1, 4, 6, 8, 6, 1, 2, 0, 1, 7, 1, 2, 3, 9, 3, 7, 2, 3, 3, 2, 0, 5,
                                       8, 6, 9, 4, 9, 5, 2, 9, 0, 7, 2, 1, 7, 9, 7, 1, 4, 8, 3, 3, 7, 0, 6, 1,
                                       4, 9, 9, 1, 5, 6, 0, 5, 2, 6, 6, 7, 7, 1, 6, 3, 7, 9, 8, 3, 2, 2, 6, 9,
                                       0, 1, 2, 9, 6, 6, 7, 0, 0, 2, 9, 0, 9, 6, 4, 0, 1, 1, 0, 1, 6, 1, 8, 2,
                                       7, 8, 1, 8, 5, 7, 4, 0]))

    first_image = input_batch[0][0]

    # upper left pixel should be normalized, and black.
    assert (first_image[0][0] - (-0.4242)) < 0.0001


if __name__ == '__main__':
    test_deterministic_dataset()
