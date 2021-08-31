import numpy as np

import main
from critic import Critic
from explainer import Explainer

from config import critic_only_config as ct_cfg
from config import classification_only_cfg as ex_cfg


def test_critic_makes_progress_without_explanations():
    *_, critic_loader = main.load_data(ct_cfg)
    critic = Critic(ct_cfg)
    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch=0,
                                                      use_explanations=False)
    assert abs(initial_loss - np.log(len(ct_cfg.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03


def test_explainer_makes_progress_with_only_classification():
    train_loader, *_ = main.load_data(ex_cfg)
    explainer = Explainer(ex_cfg)
    initial_loss, end_of_training_loss = explainer.train(train_loader, use_critic=False)

    assert abs(initial_loss - np.log(len(ex_cfg.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03
