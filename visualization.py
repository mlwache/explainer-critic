# from typing import Any, Iterator
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional

from config import default_config as cfg


class Visualizer:

    # function to show an image
    @staticmethod
    def image_show(images):
        combined_image = torchvision.utils.make_grid(images)
        inverted = functional.invert(combined_image)
        # img_un_normalized = image / 2 + 0.5
        np_img = inverted.numpy()  # img_un_normalized.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))  # .astype(np.uint8)*255)  # added `.astype(np.uint8)*255` here
        # to get rid of "clipping Data to valid range" error, but it's less easy with np.float, as this doesn't have
        # guarantees on the range. However the quality is a bit worse this way, maybe I should change it back some time.
        # But I don't think so, as it's only for the visualization.
        # (https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa)
        plt.show()

    @staticmethod
    def show_some_sample_images(images, labels):
        # show images
        Visualizer.image_show(images)
        # print labels
        print('labels: ', ' '.join('%5s' % cfg.classes[labels[j]] for j in range(cfg.batch_size)))

    @staticmethod
    def amplify_and_show(images: Tensor):
        amplified_images = torch.empty_like(images)
        for i, img in enumerate(images):
            maximum_brightness = torch.max(img)
            minimum_brightness = torch.min(img)
            maximum_value = max(abs(maximum_brightness), abs(minimum_brightness))
            # divide by the highest absolute value, so that the resulting values are all in [-1,1] :
            amplified_images[i] = torch.div(img, maximum_value)
        Visualizer.image_show(amplified_images)
