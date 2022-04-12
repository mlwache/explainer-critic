from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import global_vars
import utils
from explainer import Explainer


def run_console_evaluation_experiments():
    critic_comparison("input")


def set_up_evaluation_experiments(n_models: int,
                                  run_name: Optional[str] = None,
                                  loaders=None,
                                  used_for_training=False,
                                  n_test_samples=10000,
                                  batch_size=100
                                  ) -> Tuple[List[Explainer],
                                             DataLoader,
                                             List[str]]:
    args = utils.parse_args(overriding_args=[f'--run_name={run_name}'])
    utils.setup(args, eval_mode=not used_for_training)

    model_paths = ["trained_explainer.pt",
                   "pre-trained.pt"]

    explanation_modes = ["input_x_gradient",
                         "input",
                         "nothing"][0:n_models]
    explainers: List[Explainer] = get_list_of_empty_explainers(explanation_modes=explanation_modes,
                                                               loaders=loaders)
    for i in range(n_models):
        if i < len(model_paths):
            explainers[i].load_state(f"models/{model_paths[i]}")
        else:
            print("Not enough models paths specified, using un-trained model instead.")
            model_paths.append("un-trained.pt")
        explainers[i].classifier.eval()

    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1,
                              n_test_samples=n_test_samples,
                              batch_size=100,
                              test_batch_size=batch_size)

    return explainers, loaders.test, model_paths


def get_list_of_empty_explainers(explanation_modes, loaders) -> List[Explainer]:
    return [Explainer(loaders=loaders,
                      optimizer_type=None,
                      test_batch_to_visualize=None,
                      model_path="",
                      explanation_mode=explanation_mode) for explanation_mode in explanation_modes]


def critic_comparison(mode: str):
    """compares training the critic on the explainer's explanations to the critic learning the task itself.

    Learning the task itself just meaning classification on the input"""

    global_vars.global_step = 0

    utils.set_device()
    loaders = utils.load_data(n_training_samples=1,
                              n_critic_samples=1000,
                              n_test_samples=1000,
                              batch_size=10,
                              test_batch_size=10)
    explainers, *_ = set_up_evaluation_experiments(n_models=1,
                                                   used_for_training=True,
                                                   loaders=loaders,
                                                   run_name=f"critic-comparison-{mode}")

    explainer = explainers[0]

    explainer.train_critic_on_explanations(critic_lr=0.2, shuffle_critic=True, explanation_mode=mode)

    test_explanations = explainer.get_labeled_explanation_batches(loaders.test)
    critic_accuracy = utils.compute_accuracy(explainer.critic.classifier, test_explanations)
    print(f"Critic Accuracy on {mode}: {critic_accuracy} ")
    # TODO: plot accuracy during training for better visibility


def variance(points: Tensor) -> float:
    """computes the variance of a cluster of points"""
    mean_point = torch.mean(points, dim=0)
    differences_to_mean = points - mean_point
    differences_as_vector = torch.flatten(differences_to_mean, start_dim=1, end_dim=3)
    differences_square = differences_as_vector * differences_as_vector
    l_2_distances = torch.sqrt(differences_square.sum(dim=1))
    return torch.mean(l_2_distances).item()


def intra_class_variances(inputs: Tensor, labels: Tensor) -> List[float]:
    """sorts the points by their labels, and returns a list of the variances by label """
    contained_labels = torch.unique(labels).tolist()
    if len(contained_labels) != 10:
        print("Warning: not all labels are represented in the data!")

    intraclass_variances: List[float] = []
    for label in contained_labels:
        label_subset = inputs[labels == label]
        intraclass_variances.append(variance(label_subset))
    return intraclass_variances


if __name__ == "__main__":
    run_console_evaluation_experiments()
