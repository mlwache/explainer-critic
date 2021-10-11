# from typing import Any, Iterator
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional

# from torchviz import make_dot
from utils import colored


class ImageHandler:

    MEAN_MNIST: float
    STD_DEV_MNIST: float

    device: str
    writer: Optional[SummaryWriter]
    # function to show an image

    @staticmethod
    def show_images(images: Tensor):
        # assume image is scaled to [0,1]
        assert 0 <= images.max() <= 1
        assert 0 <= images.min() <= 1
        combined_image = torchvision.utils.make_grid(images)
        # invert image for better visibility
        inverted = functional.invert(combined_image)
        np_img = inverted.cpu().numpy()  # img_un_normalized.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))  # .astype(np.uint8)*255)  # added `.astype(np.uint8)*255` here
        # to get rid of "clipping Data to valid range" error, but it's less easy with np.float, as this doesn't have
        # guarantees on the range. However the quality is a bit worse this way, maybe I should change it back some time.
        # But I don't think so, as it's only for the visualization.
        # (https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa)
        plt.show()

    @staticmethod
    def rescale_and_show(images: Tensor):
        amplified_images = ImageHandler.rescale_to_zero_one(images)
        ImageHandler.show_images(amplified_images)

    @staticmethod
    def rescale_to_zero_one(images: Tensor):
        # detach in order to be able to process like an ndarray.
        images = abs(images)
        amplified_images: Tensor = images / images.max()
        return amplified_images

#     @staticmethod
#     def show_computation_graph(labels, parameters):
#         raise NotImplementedError
# #        make_dot(y, parameters)

    @staticmethod
    def add_input_images(input_images: Tensor, caption: str = "some input images"):
        input_images = input_images * ImageHandler.STD_DEV_MNIST + ImageHandler.MEAN_MNIST  # un-normalize
        print('Adding input images.')
        ImageHandler.add_image_grid_to_writer(caption, input_images)

    @staticmethod
    def add_gradient_images(test_batch: Tuple[Tensor, Tensor], explainer, additional_caption: str):
        test_images, test_labels = test_batch
        rescaled_input_gradient: Tensor = explainer.rescaled_input_gradient(test_images, test_labels)

        ImageHandler.add_image_grid_to_writer(f"gradient/{additional_caption}", rescaled_input_gradient)

        grad_x_input = ImageHandler.rescale_to_zero_one(rescaled_input_gradient*test_images)
        ImageHandler.add_image_grid_to_writer(f"gradient x input/{additional_caption}", grad_x_input)

    @staticmethod
    def add_image_grid_to_writer(caption: str, some_images: Tensor):
        combined_image = torchvision.utils.make_grid(some_images)
        if ImageHandler.writer:
            ImageHandler.writer.add_image(caption, combined_image)
        else:
            print(colored(200, 0, 0, f"No writer set - skipped adding {caption} images."))
