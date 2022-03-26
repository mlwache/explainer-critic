import argparse
from typing import List

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import global_vars
import utils
from evaluation_experiments_web import set_up_evaluation_experiments
from explainer import Explainer


def run_console_evaluation_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["input_x_gradient", "input"])
    args = parser.parse_args()

    get_importance_maps()
    # critic_comparison(args.mode)


def critic_comparison(mode: str):
    """compares training the critic on the explainer's explanations to the critic learning the task itself.

    Learning the task itself just meaning classification on the input"""

    global_vars.global_step = 0

    explainers: List[Explainer]
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


def get_importance_maps():
    importance_map_set_train = ImportanceMapMNIST(root="./data", train=True)
    importance_map_set_test = ImportanceMapMNIST(root="./data", train=False)


class ImportanceMapMNIST(MNIST):

    def __init__(self, root: str, train: bool, download: bool = True):
        super().__init__(root=root, train=train, download=download)

        # Put both data and targets on GPU in advance
        device = utils.get_device()
        self.data, self.targets = self.data.to(device), self.targets.to(device)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Transform data to also contain the importance maps
        self.augment_data_with_importance_maps()

    def augment_data_with_importance_maps(self):
        """compute importance map for each image, and add it to self.data. Todo."""

        # self.data = self.compute_importance_maps()

    def compute_importance_maps(self):
        input_loader = DataLoader(self.data)
        # TODO: transform each image to an importance map
        # this should be relatively easy to do in parallel.
        importance_maps = self.data
        return importance_maps

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


if __name__ == "__main__":
    run_console_evaluation_experiments()
