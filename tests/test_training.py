import numpy as np

import main
from critic import Critic

from config import critic_test_config as ct_cfg


def test_critic_makes_progress_without_explanations():
    *_, critic_loader = main.load_data(ct_cfg)
    critic = Critic(ct_cfg)
    initial_loss, end_of_training_loss = critic.train(critic_loader, explanations=[], n_explainer_batch=0,
                                                      use_explanations=False)
    assert abs(initial_loss - np.log(10)) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03
