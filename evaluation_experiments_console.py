import argparse
from typing import List

import global_vars
import utils
from evaluation_experiments_web import set_up_evaluation_experiments
from explainer import Explainer


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


def run_console_evaluation_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["input_x_gradient", "input"])
    args = parser.parse_args()

    critic_comparison(args.mode)


if __name__ == "__main__":
    run_console_evaluation_experiments()
