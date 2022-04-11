
import numpy as np
import pytest
import torch

import utils
from config import SimpleArgumentParser
from experiments import train_only_critic, run_experiments
from explainer import Explainer
from evaluation_experiments_console import variance, intra_class_variances


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
    initial_loss, end_of_training_loss = train_only_critic(args.n_critic_batches, args.batch_size,
                                                           args.learning_rate_critic, explanations=[])
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.1


def test_explainer_makes_progress_with_only_classification(args):
    n_classes = 10
    n_training_samples = 20 * args.batch_size  # 20 batches
    loaders = utils.load_data(n_training_samples=n_training_samples,
                              n_critic_samples=1,
                              n_test_samples=1,
                              batch_size=args.batch_size,
                              test_batch_size=1)
    utils.set_device()
    explainer = Explainer(loaders, args.optimizer, test_batch_to_visualize=None,
                          model_path="", explanation_mode=args.explanation_mode)
    initial_loss, end_of_training_loss = explainer.pretrain(args.pretrain_learning_rate, args.learning_rate_step,
                                                            lr_scheduling=False, n_epochs=1)
    assert abs(initial_loss - np.log(n_classes)) < 0.1
    assert initial_loss - end_of_training_loss > 0.1


def test_variance():
    tensor1 = torch.Tensor([[0, 1], [2, 3]])
    tensor2 = torch.Tensor([[0, 2], [4, 6]])
    tensor1 = torch.unsqueeze(tensor1, dim=2)
    tensor2 = torch.unsqueeze(tensor2, dim=2)
    variance1 = variance(torch.stack([tensor1, tensor1]))
    assert variance1 == 0
    variance2 = variance(torch.stack([tensor1, tensor2]))
    assert abs(variance2 - 1.87083) < 0.01


def test_intra_class_variance_simple_tensors():
    tensor1 = torch.Tensor([[0, 1], [2, 3]])
    tensor2 = torch.Tensor([[0, 2], [4, 6]])
    tensor1 = torch.unsqueeze(tensor1, dim=2)
    tensor2 = torch.unsqueeze(tensor2, dim=2)
    inputs1 = torch.stack([tensor1, tensor2, tensor1, tensor2])
    inputs2 = torch.stack([tensor1, tensor1, tensor2, tensor2])
    labels = torch.tensor([0, 0, 1, 1])
    assert variance(inputs1) == variance(inputs2)
    total_var = variance(inputs1)
    assert intra_class_variances(inputs1, labels) == [total_var, total_var]
    assert intra_class_variances(inputs2, labels) == [0, 0]


def test_git_root():
    git_root = utils.get_git_root()
    # no assert because it's different for every device
    print(f"git root: {git_root}")
