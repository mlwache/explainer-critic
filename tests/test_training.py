import numpy as np

from experiments.experiments import train_critic_without_explanations, train_explainer_only_classification

from config import default_config as cfg


def test_critic_makes_progress_without_explanations():
    initial_loss, end_of_training_loss = train_critic_without_explanations(tensorboard_enabled=False)
    assert abs(initial_loss - np.log(len(cfg.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03


def test_explainer_makes_progress_with_only_classification():
    initial_loss, end_of_training_loss = train_explainer_only_classification(tensorboard_enabled=False)
    assert abs(initial_loss - np.log(len(cfg.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03
