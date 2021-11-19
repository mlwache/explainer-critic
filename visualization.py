from typing import Tuple, Optional

import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class ImageHandler:

    MEAN_MNIST: float = 0.1307
    STD_DEV_MNIST: float = 0.3081

    device: str
    writer: Optional[SummaryWriter]

    @staticmethod
    def rescale_to_zero_one(images: Tensor):
        assert (images >= 0).all(), "it's expected that all values are non-negative."
        amplified_images: Tensor = images / images.max()
        return amplified_images

    @staticmethod
    def add_input_images(input_images: Tensor, caption: str = "some input images"):
        input_images = input_images * ImageHandler.STD_DEV_MNIST + ImageHandler.MEAN_MNIST  # un-normalize
        print('Adding input images.')
        # global step is zero, as this is only done once.
        ImageHandler.add_image_grid_to_writer(caption, input_images)

    @staticmethod
    def add_gradient_images(test_batch: Tuple[Tensor, Tensor], explainer, additional_caption: str,
                            global_step: int = None):
        test_images, test_labels = test_batch

        rescaled_explanation_batch: Tensor = explainer.get_explanation_batch(test_images, test_labels)

        # if the combination is multiplication, show only the gradient as well
        if explainer.explanation_mode == "input_x_gradient" or "input_x_integrated_gradient":
            gradient = rescaled_explanation_batch/test_images
            ImageHandler.add_image_grid_to_writer(caption=f"(integrated) gradient/{additional_caption}",
                                                  some_images=gradient,
                                                  global_step=global_step)
            # un-normalizing is also only necessary if we are in input-space.
            explanation_batch_show = ImageHandler.un_normalize(rescaled_explanation_batch)
        else:
            explanation_batch_show = rescaled_explanation_batch

        ImageHandler.add_image_grid_to_writer(caption=f"{explainer.explanation_mode}/{additional_caption}",
                                              some_images=explanation_batch_show,
                                              global_step=global_step)

    @staticmethod
    def add_image_grid_to_writer(caption: str, some_images: Tensor, global_step: int = None):
        combined_image = torchvision.utils.make_grid(some_images)
        if ImageHandler.writer:
            ImageHandler.writer.add_image(caption, combined_image, global_step=global_step)
        else:
            print(colored(200, 0, 0, f"No writer set - skipped adding {caption} images."))

    @staticmethod
    def un_normalize(images):
        return images.mul_(ImageHandler.STD_DEV_MNIST).add_(ImageHandler.MEAN_MNIST)


def colored(r, g, b, text):
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
