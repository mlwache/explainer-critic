from typing import Tuple, Any

from torch.utils.data import DataLoader

import main
from config import SimpleArgumentParser
from critic import Critic
from explainer import Explainer

Loss = float


def run_experiments():
    print("Setting up experiments...")
    args = SimpleArgumentParser()
    args.parse_args()
    train_loader, critic_loader, explainer = set_up_experiments_combined(args)

    print("Training together with simple combined loss...")
    print(f"initial/final loss:{train_together(explainer, critic_loader, train_loader)}")
    explainer.reset()
    print("Now, see what happens when we pre-train the explainer first...")
    print(f"initial/final loss (pretraining): {explainer.pre_train(train_loader)}")
    explainer.set_writer_step_offset(len(train_loader), len(critic_loader))
    print(f"initial/final loss (together, after pretraining):{train_together(explainer, critic_loader, train_loader)}")
    print(f"initial/final loss (only critic): {train_critic_without_explanations(args)}")
    print(f"initial/final loss (only classification): {train_explainer_only_classification(args)}")
    train_in_turns()


def set_up_experiments_combined(args: SimpleArgumentParser) -> Tuple[DataLoader[Any], DataLoader[Any], Explainer]:
    train_loader, _, critic_loader = main.load_data(args)
    explainer = Explainer(args)
    return train_loader, critic_loader, explainer


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
    run_experiments()
