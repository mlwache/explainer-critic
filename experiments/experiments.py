from typing import Tuple

import main
from config import SimpleArgumentParser
from critic import Critic
from explainer import Explainer

Loss = float


def run_experiments():
    args = SimpleArgumentParser()
    args.parse_args()
    train_critic_without_explanations(args)
    train_explainer_only_classification(args)
    train_in_turns()


def train_critic_without_explanations(args: SimpleArgumentParser) -> Tuple[Loss, Loss]:
    args.batch_size = 128
    args.n_critic_batches = 100
    *_, critic_loader = main.load_data(args)
    critic = Critic(args)
    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch=0)
    return initial_loss, end_of_training_loss


def train_explainer_only_classification(args: SimpleArgumentParser) -> Tuple[Loss, Loss]:
    args.batch_size = 128
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
