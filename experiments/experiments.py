from typing import Tuple

import main
from critic import Critic
from explainer import Explainer

from config import critic_only_config as ct_cfg
from config import classification_only_cfg as ex_cfg
Loss = float


def run_experiments():
    train_critic_without_explanations()
    train_explainer_only_classification()
    train_in_turns()


def train_critic_without_explanations(tensorboard_enabled=True) -> Tuple[Loss, Loss]:
    *_, critic_loader = main.load_data(ct_cfg)
    if tensorboard_enabled:
        main.make_tensorboard_writer(ct_cfg)
    critic = Critic(ct_cfg)
    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch=0)
    return initial_loss, end_of_training_loss


def train_explainer_only_classification(tensorboard_enabled=True) -> Tuple[Loss, Loss]:
    train_loader, *_ = main.load_data(ex_cfg)
    if tensorboard_enabled:
        main.make_tensorboard_writer(ex_cfg)
    explainer = Explainer(ex_cfg)
    initial_loss, end_of_training_loss = explainer.train(train_loader, use_critic=False)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments()
