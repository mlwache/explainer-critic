from typing import Tuple, List

from torch.utils.tensorboard import SummaryWriter

import global_vars
import main
import utils
from config import SimpleArgumentParser
from critic import Critic
from explainer import Explainer
from utils import Loaders
from visualization import ImageHandler

Loss = float


def run_experiments(optional_args: List):
    global_vars.global_step = 0
    print("Setting up experiments...")
    loaders, args, device, writer = set_up_experiments_combined(optional_args)

    test_batch_to_visualize = utils.get_one_batch_of_images(device, loaders.visualization)
    explainer = Explainer(args, device, test_batch_to_visualize, writer)
    ImageHandler.add_input_images(test_batch_to_visualize[0])
    ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="0: before training")

    if args.training_mode == "combined":
        print("Training together with simple combined loss...")
        init_l, fin_l = train_together(explainer, loaders, args.log_interval)
        print(f"initial/final loss:{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrain_from_scratch":
        print("Pre-train the explainer first...")
        init_l_p, fin_l_p = explainer.pre_train(loaders.train, loaders.test)
        print(f"initial/final loss (pretraining):{init_l_p:.3f}, {fin_l_p:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_l, fin_l = train_together(explainer, loaders, args.log_interval)
        print(f"initial/final loss (combined, after pretraining):{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrained":
        explainer.load_state("./models/pretrained_model.pt")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_l, fin_l = train_together(explainer, loaders, args.log_interval)
        print(f"initial/final loss (combined, after pretraining):{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "only_critic":
        print(utils.colored(200, 0, 0, "Only training critic, progress output may still be buggy."))
        init_l, fin_l = train_only_critic(args, device, explanations=[])
        print(f"initial/final loss (only critic): {init_l}, {fin_l}")

    elif args.training_mode == "only_classification":
        init_l_p, fin_l_p = explainer.pre_train(loaders.train, loaders.test)
        print(f"initial/final loss (only classification): {init_l_p}, {fin_l_p}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after only-classification training")
        print(f"initial/final loss (pretraining):{init_l_p:.3f}, {fin_l_p:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")

    elif args.training_mode == "in_turns":
        train_in_turns()
    elif args.training_mode == "one_critic_pass":
        init_l_p, fin_l_p = explainer.pre_train(loaders.train, loaders.test)
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        fin_l_p = explainer.explanation_loss(loaders.critic)
        print(f"initial/final loss (one critic pass): {init_l_p}, {fin_l_p}")
    else:
        raise ValueError(f'Invalid training mode "{args.training_mode}"!')


def set_up_experiments_combined(optional_args: List) -> Tuple[Loaders, SimpleArgumentParser, str, SummaryWriter]:
    args, device, writer = main.setup(optional_args)
    loaders = utils.load_data_from_args(args)
    return loaders, args, device, writer


def train_together(explainer: Explainer, loaders: Loaders, log_interval: int) -> Tuple[Loss, Loss]:
    return explainer.train(loaders.train, loaders.critic, loaders.test, log_interval)


def train_only_critic(args: SimpleArgumentParser, device: str, explanations: List) -> Tuple[Loss, Loss]:
    critic = Critic(args, device)

    loaders = utils.load_data(n_training_samples=1, n_critic_samples=args.n_critic_batches * args.batch_size,
                              n_test_samples=1, batch_size=args.batch_size)

    initial_loss, end_of_training_loss = critic.train(loaders.critic, explanations)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments([])
