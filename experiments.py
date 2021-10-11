from typing import Tuple, Any, List

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import main
import utils
from visualization import ImageHandler
from config import SimpleArgumentParser
from critic import Critic
from explainer import Explainer

Loss = float


def run_experiments(optional_args: List):
    print("Setting up experiments...")
    train_loader, critic_loader, test_loader, args, device, writer \
        = set_up_experiments_combined(optional_args)

    test_batch_to_visualize = utils.get_one_batch_of_images(device, test_loader)
    explainer = Explainer(args, device, test_batch_to_visualize, writer)
    ImageHandler.add_input_images(test_batch_to_visualize[0])
    ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="before training")

    if args.training_mode == "combined":
        print("Training together with simple combined loss...")
        init_l, fin_l = train_together(explainer, critic_loader, train_loader, test_loader, args.log_interval)
        print(f"initial/final loss:{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after combined training")

    elif args.training_mode == "pretrain":
        print("Pre-train the explainer first...")
        init_l_p, fin_l_p = explainer.pre_train(train_loader, test_loader, args.n_pretraining_epochs,
                                                log_interval=args.log_interval_pretraining)
        print(f"initial/final loss (pretraining):{init_l_p:.3f}, {fin_l_p:3f}")
        # explainer.set_pretraining_writer_step_offset(pre_training_set_size=len(train_loader),
        #                                              critic_set_size=len(critic_loader))
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="after pretraining")
        init_l, fin_l = train_together(explainer, critic_loader, train_loader, test_loader, args.log_interval)
        print(f"initial/final loss (combined, after pretraining):{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after combined training")

    elif args.training_mode == "only_critic":
        print(utils.colored(200, 0, 0, "Only training critic, progress output may still be buggy."))
        init_l, fin_l = train_critic_without_explanations(args, device)
        print(f"initial/final loss (only critic): {init_l}, {fin_l}")

    elif args.training_mode == "only_classification":
        init_l_p, fin_l_p = train_explainer_only_classification(args, device, train_loader, test_loader)
        print(f"initial/final loss (only classification): {init_l_p}, {fin_l_p}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after only-classification training")

    elif args.training_mode == "in_turns":
        train_in_turns()
    else:
        raise ValueError(f'Invalid training mode "{args.training_mode}"!')


def set_up_experiments_combined(optional_args: List) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any],
                                                              SimpleArgumentParser, str, SummaryWriter]:
    args, device, writer = main.setup(optional_args, "experiments")
    train_loader, test_loader, critic_loader = utils.load_data_from_args(args)
    return train_loader, critic_loader, test_loader, args, device, writer


def train_together(explainer: Explainer, critic_loader: DataLoader[Any], train_loader: DataLoader[Any],
                   test_loader: DataLoader[Any], log_interval: int) -> Tuple[Loss, Loss]:
    return explainer.train(train_loader, critic_loader, test_loader, log_interval)


def train_critic_without_explanations(args: SimpleArgumentParser, device: str) -> Tuple[Loss, Loss]:
    critic = Critic(args, device)

    *_, critic_loader = utils.load_data(n_training_samples=1, n_critic_samples=args.n_critic_batches * args.batch_size,
                                        n_test_samples=1, batch_size=args.batch_size)

    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch_total=0)
    return initial_loss, end_of_training_loss


def train_explainer_only_classification(args: SimpleArgumentParser, device: str, train_loader: DataLoader[Any],
                                        test_loader: DataLoader[Any]) -> Tuple[Loss, Loss]:
    args.n_training_batches = 100
    explainer = Explainer(args, device)
    initial_loss, end_of_training_loss = explainer.pre_train(train_loader, test_loader, n_epochs=1)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments([])
