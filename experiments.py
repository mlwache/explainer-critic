from typing import Tuple, Any, List

from torch.utils.data import DataLoader

import main
from visualization import ImageHandler
from config import SimpleArgumentParser
from critic import Critic
from explainer import Explainer

Loss = float


def run_experiments(optional_args: List):
    print("Setting up experiments...")
    train_loader, critic_loader, explainer, args = set_up_experiments_combined(optional_args)

    ImageHandler.show_batch(args, train_loader, explainer, additional_caption="before training")
    if args.training_mode == "combined":
        print("Training together with simple combined loss...")
        print(f"initial/final loss:{train_together(explainer, critic_loader, train_loader)}")
        ImageHandler.show_batch(args, train_loader, explainer, additional_caption="after combined training")
    elif args.training_mode == "pretrain":
        print("See what happens when we pre-train the explainer first...")
        print(f"initial/final loss (pretraining): {explainer.pre_train(train_loader, args.n_pretraining_epochs)}")
        # explainer.set_pretraining_writer_step_offset(pre_training_set_size=len(train_loader),
        #                                              critic_set_size=len(critic_loader))
        ImageHandler.show_batch(args, train_loader, explainer, additional_caption="after pretraining")
        print(f"initial/final loss (together, after pretraining):")
        print(f"{train_together(explainer, critic_loader, train_loader)}")
        ImageHandler.show_batch(args, train_loader, explainer, additional_caption="after combined training")
    elif args.training_mode == "only_critic":
        print(f"initial/final loss (only critic): {train_critic_without_explanations(args)}")
    elif args.training_mode == "only_classification":
        print(f"initial/final loss (only classification): {train_explainer_only_classification(args)}")
        ImageHandler.show_batch(args, train_loader, explainer, additional_caption="after only-classification training")
    elif args.training_mode == "in_turns":
        train_in_turns()
    else:
        print(f'Invalid training mode "{args.training_mode}"!')


def set_up_experiments_combined(optional_args: List) -> Tuple[DataLoader[Any], DataLoader[Any], Explainer,
                                                              SimpleArgumentParser]:
    args = main.setup(optional_args)
    train_loader, _, critic_loader = main.load_data(args)
    explainer = Explainer(args)
    return train_loader, critic_loader, explainer, args


def train_together(explainer: Explainer, critic_loader: DataLoader[Any], train_loader: DataLoader[Any]):
    explainer.train(train_loader, critic_loader)


def train_critic_without_explanations(args: SimpleArgumentParser) -> Tuple[Loss, Loss]:
    args.n_critic_batches = 100
    *_, critic_loader = main.load_data(args)
    critic = Critic(args)
    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch=0)
    return initial_loss, end_of_training_loss


def train_explainer_only_classification(args: SimpleArgumentParser) -> Tuple[Loss, Loss]:
    args.n_training_batches = 100
    args.n_critic_batches = 0
    train_loader, *_ = main.load_data(args)
    explainer = Explainer(args)
    initial_loss, end_of_training_loss = explainer.train(train_loader, use_critic=False)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments([])
