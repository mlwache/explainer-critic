import numpy as np
import pytest

import utils
from config import SimpleArgumentParser
from experiments import train_only_critic, run_experiments
from explainer import Explainer


@pytest.fixture
def args() -> SimpleArgumentParser:
    args = SimpleArgumentParser()
    args.parse_args(args=['--logging_disabled', '--batch_size=32', '--n_epochs=1'])
    return args


def test_experiments_dont_crash():
    for training_mode in ["pretrained", "combined", "only_critic", "only_classification", "one_critic_pass",
                          "pretrain_from_scratch"]:
        # Todo: "in_turns"

        for explanation_mode in ["input_x_gradient", "gradient", "integrated_gradient"]:
            run_experiments(['--batch_size=4', '--n_training_batches=2', '--n_critic_batches=2', '--n_test_batches=1',
                             '--n_epochs=2', '--logging_disabled', '--n_pretraining_epochs=1',
                             f'--training_mode={training_mode}', f'--explanation_mode={explanation_mode}'])


def test_critic_makes_progress_without_explanations(args: SimpleArgumentParser):
    n_classes = 10
    args.n_critic_batches = 50
    initial_loss, end_of_training_loss = train_only_critic(utils.get_device(), args.n_critic_batches, args.batch_size,
                                                           args.learning_rate_critic, explanations=[])
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.1


def test_explainer_makes_progress_with_only_classification(args):
    n_classes = 10
    n_training_samples = 20 * args.batch_size  # 20 batches
    loaders = utils.load_data(n_training_samples, n_critic_samples=0, n_test_samples=0,
                              batch_size=args.batch_size)
    explainer = Explainer(utils.get_device(), loaders, args.optimizer, logging=None, test_batch_to_visualize=None,
                          rtpt=None, model_path="", explanation_mode=args.explanation_mode)
    initial_loss, end_of_training_loss = explainer.pretrain(args.pretrain_learning_rate, args.learning_rate_step,
                                                            lr_scheduling=False, n_epochs=1)
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.1
