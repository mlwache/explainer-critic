# imports necessary for Tensorboard Logging
from typing import Tuple, List

# torch related imports
from torch.utils.tensorboard import SummaryWriter

# local imports
from config import SimpleArgumentParser
from explainer import Explainer
from utils import get_device, set_sharing_strategy, write_config_to_log, config_string, colored, \
    load_data_from_args
from visualization import ImageHandler


def main():
    args, device, writer = setup([])

    print('Loading Data...')
    train_loader, test_loader, critic_loader = load_data_from_args(args)
    explainer = Explainer(args, device, writer)

    if args.rtpt_enabled:
        explainer.start_rtpt(args.n_training_batches)

    print(f'Training the Explainer on {args.n_training_samples} samples...')
    explainer.train(train_loader, critic_loader, test_loader, args.log_interval)

    explainer_accuracy = explainer.compute_accuracy(test_loader)
    print(f'Explainer Accuracy on {args.n_test_samples} test images: {100 * explainer_accuracy} %')


def setup(optional_args: List) -> Tuple[SimpleArgumentParser, str, SummaryWriter]:
    args = SimpleArgumentParser()
    if optional_args:
        args.parse_args(optional_args)
    else:
        args.parse_args()

    set_sharing_strategy()

    if not args.logging_disabled:
        log_dir = f"./runs/{config_string(args)}"
        write_config_to_log(args, log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    device = get_device()

    ImageHandler.device, ImageHandler.writer = device, writer
    ImageHandler.MEAN_MNIST, ImageHandler.STD_DEV_MNIST = args.MEAN_MNIST, args.STD_DEV_MNIST

    return args, device, writer


if __name__ == '__main__':
    main()
    print(colored(0, 200, 0, "Finished!"))
