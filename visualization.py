# from typing import Any, Iterator
from typing import Any, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import functional

# from torchviz import make_dot


class ImageHandler:

    device: str = ""
    writer: Optional[SummaryWriter] = None
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
        amplified_images = ImageHandler.re_scale_to_zero_one(images)
        ImageHandler.show_images(amplified_images)

    @staticmethod
    def re_scale_to_zero_one(images: Tensor):
        # detach in order to be able to process like an ndarray.
        images = abs(images)
        amplified_images: Tensor = images / images.max()
        return amplified_images

#     @staticmethod
#     def show_computation_graph(labels, parameters):
#         raise NotImplementedError
# #        make_dot(y, parameters)
    @staticmethod
    def show_batch(args, loader: DataLoader[Any], explainer, additional_caption: str = ""):
        some_images, some_labels = ImageHandler.get_one_batch_of_images(loader)
        print('Here are some images!')
        ImageHandler.visualize_input(args, some_images, additional_caption)
        print('And their gradients!')
        ImageHandler.visualize_gradient(args, explainer, some_images, some_labels, additional_caption)

    @staticmethod
    def visualize_gradient(args, explainer, some_test_images, some_test_labels, additional_caption: str = ""):
        rescaled_input_gradient: Tensor = explainer.rescaled_input_gradient(some_test_images, some_test_labels)
        # ImageHandler.show_images(rescaled_input_gradient)

        if ImageHandler.writer:
            ImageHandler.add_image_grid_to_writer("gradient", additional_caption, ImageHandler.writer,
                                                  rescaled_input_gradient)

        grad_x_input = (rescaled_input_gradient*some_test_images) * args.STD_DEV_MNIST + args.MEAN_MNIST
        if ImageHandler.writer:
            ImageHandler.add_image_grid_to_writer("gradient x input", additional_caption, ImageHandler.writer,
                                                  grad_x_input)

        # grid_grad = torchvision.utils.make_grid(rescaled_input_gradient)
        # caption = "gradient, " + additional_caption
        # args.WRITER.add_image(caption, grid_grad)

    @staticmethod
    def visualize_input(args, some_images, additional_caption: str = ""):
        some_images = some_images * args.STD_DEV_MNIST + args.MEAN_MNIST
        # ImageHandler.show_images(some_images)
        if ImageHandler.writer:
            ImageHandler.add_image_grid_to_writer("some input images", additional_caption, ImageHandler.writer,
                                                  some_images)

    @staticmethod
    def add_image_grid_to_writer(image_type_caption: str, additional_caption: str, writer, some_images):
        combined_image = torchvision.utils.make_grid(some_images)
        caption = image_type_caption + ", " + additional_caption
        writer.add_image(caption, combined_image)

    @staticmethod
    def get_one_batch_of_images(loader: DataLoader[Any]) -> Tuple[Tensor, Tensor]:
        images, labels = next(iter(loader))
        images, labels = images.to(ImageHandler.device)[:4], labels.to(ImageHandler.device)[:4]
        return images, labels
