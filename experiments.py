from typing import Tuple, List, Optional

import global_vars
import utils
from critic import Critic
from explainer import Explainer
from visualization import ImageHandler

Loss = float


def run_experiments(overriding_args: Optional[List] = None):

    print("Setting up experiments...")
    args = utils.parse_args(overriding_args)
    utils.setup(args)
    # start at the negative pretraining iterations, so the logging of combined training starts at step zero.
    global_vars.global_step = -(args.n_iterations - args.combined_iterations)
    loaders = utils.load_data_from_args(args)

    test_batch_to_visualize = utils.get_one_batch_of_images(loaders.visualization)
    explainer = Explainer(loaders,
                          args.optimizer,
                          test_batch_to_visualize,
                          model_path=f"models/{utils.config_string(args)}.pt",
                          explanation_mode=args.explanation_mode)
    ImageHandler.add_input_images(test_batch_to_visualize[0])  # needs only the images, not the labels
    ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="0: before training")

    if args.training_mode == "combined":
        print("Training together with simple combined loss...")
        init_loss, fin_loss = explainer.train_from_args(args)
        print(f"initial/final loss:{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrain_from_scratch":
        print("Pre-train the explainer first...")
        init_loss_pretraining, final_loss_pretraining = explainer.pretrain_from_args(args)
        print(f"initial/final loss (pretraining):{init_loss_pretraining:.3f}, {final_loss_pretraining:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_loss, fin_loss = explainer.train_from_args(args)
        print(f"initial/final loss (combined, after pretraining):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "pretrained":
        explainer.load_state("./models/pretrained_model.pt")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        init_loss, fin_loss = explainer.train_from_args(args)
        print(f"initial/final loss (combined, after pretraining):{init_loss:.3f}, {fin_loss:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="3: after combined training")

    elif args.training_mode == "only_critic":
        print(utils.colored(200, 0, 0, "Only training critic, progress output may still be buggy."))
        init_loss, fin_loss = train_only_critic(args.n_critic_batches, args.batch_size, args.learning_rate_critic,
                                          explanations=[])
        print(f"initial/final loss (only critic): {init_loss}, {fin_loss}")

    elif args.training_mode == "only_classification":
        init_loss_pretraining, final_loss_pretraining = explainer.pretrain_from_args(args)
        print(f"initial/final loss (only classification): {init_loss_pretraining}, {final_loss_pretraining}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer,
                                         additional_caption="after only-classification training")
        print(f"initial/final loss (pretraining):{init_loss_pretraining:.3f}, {final_loss_pretraining:3f}")
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")

    elif args.training_mode == "in_turns":
        train_in_turns()
    elif args.training_mode == "one_critic_pass":
        init_loss_pretraining, final_loss_pretraining = explainer.pretrain_from_args(args)
        ImageHandler.add_gradient_images(test_batch_to_visualize, explainer, additional_caption="1: after pretraining")
        final_loss_pretraining = explainer.train_critic_on_explanations(critic_lr=args.learning_rate_critic,
                                                         shuffle_critic=not args.disable_critic_shuffling)
        print(f"initial/mean loss (one critic pass): {init_loss_pretraining}, {final_loss_pretraining}")
    else:
        raise ValueError(f'Invalid training mode "{args.training_mode}"!')
    print(utils.colored(0, 200, 0, "Finished!"))


def train_only_critic(n_critic_batches, batch_size, critic_learning_rate,
                      explanations: List) -> Tuple[Loss, Loss]:
    global_vars.global_step = 0
    critic_loader = utils.load_data(n_training_samples=1,
                                    n_critic_samples=n_critic_batches * batch_size,
                                    n_test_samples=1,
                                    batch_size=batch_size,
                                    test_batch_size=1).critic
    critic = Critic(explanation_mode="empty",
                    critic_loader=critic_loader,
                    log_interval_critic=1,
                    shuffle_data=False)

    initial_loss, end_of_training_loss, _ = critic.train(explanations, critic_learning_rate)
    return initial_loss, end_of_training_loss


def train_in_turns():
    pass  # TODO


if __name__ == '__main__':
    run_experiments()
