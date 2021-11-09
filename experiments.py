from typing import Tuple, List, Optional

import global_vars
import utils
from critic import Critic
from explainer import Explainer
from visualization import ImageHandler

Loss = float


def run_experiments(overriding_args: Optional[List] = None):
    global_vars.global_step = 0

    print("Setting up experiments...")
    args, device, logging, rtpt = utils.setup(overriding_args)
    loaders = utils.load_data_from_args(args)

    test_batch_to_visualize = utils.get_one_batch_of_images(device, loaders.visualization)
    explainer = Explainer(device, loaders, args.optimizer, logging, test_batch_to_visualize, rtpt,
                          model_path=f"models/{utils.config_string(args)}.pt")
    ImageHandler.add_input_images(test_batch_to_visualize[0])  # needs only the images, not the labels
    ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="0: before training")

    if args.training_mode == "combined":
        print("Training together with simple combined loss...")
        init_l, fin_l = explainer.train_from_args(args)
        print(f"initial/final loss:{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrain_from_scratch":
        print("Pre-train the explainer first...")
        init_l_p, fin_l_p = explainer.pretrain_from_args(args)
        print(f"initial/final loss (pretraining):{init_l_p:.3f}, {fin_l_p:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_l, fin_l = explainer.train_from_args(args)
        print(f"initial/final loss (combined, after pretraining):{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrained":
        explainer.load_state("./models/pretrained_model.pt")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_l, fin_l = explainer.train_from_args(args)
        print(f"initial/final loss (combined, after pretraining):{init_l:.3f}, {fin_l:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "only_critic":
        print(utils.colored(200, 0, 0, "Only training critic, progress output may still be buggy."))
        init_l, fin_l = train_only_critic(device, args.n_critic_batches, args.batch_size, args.learning_rate_critic,
                                          explanations=[])
        print(f"initial/final loss (only critic): {init_l}, {fin_l}")

    elif args.training_mode == "only_classification":
        init_l_p, fin_l_p = explainer.pretrain_from_args(args)
        print(f"initial/final loss (only classification): {init_l_p}, {fin_l_p}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after only-classification training")
        print(f"initial/final loss (pretraining):{init_l_p:.3f}, {fin_l_p:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")

    elif args.training_mode == "in_turns":
        train_in_turns()
    elif args.training_mode == "one_critic_pass":
        init_l_p, fin_l_p = explainer.pretrain_from_args(args)
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        fin_l_p = explainer.train_critic_on_explanations(args.learning_rate_critic)
        print(f"initial/mean loss (one critic pass): {init_l_p}, {fin_l_p}")
    else:
        raise ValueError(f'Invalid training mode "{args.training_mode}"!')
    print(utils.colored(0, 200, 0, "Finished!"))


def train_only_critic(device: str, n_critic_batches, batch_size, critic_learning_rate,
                      explanations: List) -> Tuple[Loss, Loss]:
    critic_loader = utils.load_data(n_training_samples=1, n_critic_samples=n_critic_batches * batch_size,
                                    n_test_samples=1, batch_size=batch_size).critic
    critic = Critic(device, critic_loader, writer=None, log_interval_critic=1)

    initial_loss, end_of_training_loss, _ = critic.train(explanations, critic_learning_rate)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments()
