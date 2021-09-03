import numpy as np

from config import SimpleArgumentParser
from experiments.experiments import train_critic_without_explanations, train_explainer_only_classification


def test_critic_makes_progress_without_explanations():
    args = SimpleArgumentParser()
    args.parse_args()
    initial_loss, end_of_training_loss = train_critic_without_explanations(args)
    assert abs(initial_loss - np.log(len(args.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03


def test_explainer_makes_progress_with_only_classification():
    args = SimpleArgumentParser()
    args.parse_args()
    initial_loss, end_of_training_loss = train_explainer_only_classification(args)
    assert abs(initial_loss - np.log(len(args.CLASSES))) < 0.1
    assert end_of_training_loss < 2.28
    assert initial_loss - end_of_training_loss > 0.03
