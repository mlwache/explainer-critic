# from typing import Any, Iterator
from typing import Union

from torch import Tensor, nn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional
# from torchviz import make_dot
from config import default_config as cfg


# from explainer import Explainer
from explainer import Explainer


class Visualizer:

    # function to show an image
    @staticmethod
    def image_show(images):
        combined_image = torchvision.utils.make_grid(images)
        # invert image for better visibility
        inverted = functional.invert(combined_image)
        # img_un_normalized = image / 2 + 0.5
        np_img = inverted.cpu().numpy()  # img_un_normalized.numpy()
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
        print('labels: ', ' '.join('%5s' % cfg.CLASSES[labels[j]] for j in range(cfg.batch_size)))

    @staticmethod
    def amplify_and_show(images: Tensor):
        amplified_images = Visualizer.amplify(images)
        Visualizer.image_show(amplified_images)

    @staticmethod
    def amplify(images):
        amplified_images = torch.empty_like(images, device=cfg.DEVICE)
        for i, img in enumerate(images):
            maximum_brightness = torch.max(img)
            minimum_brightness = torch.min(img)
            maximum_value = max(abs(maximum_brightness), abs(minimum_brightness))
            # divide by the highest absolute value, so that the resulting values are all in [-1,1] :
            amplified_images[i] = torch.div(img, maximum_value)
        return amplified_images


#     @staticmethod
#     def show_computation_graph(labels, parameters):
#         raise NotImplementedError
# #        make_dot(y, parameters)

class ModelForVisualizingComputationGraph(nn.Module):
    def __init__(self, explainer, critic_loader):
        super().__init__()
        self.explainer = explainer
        self.critic_loader = critic_loader

    def forward(self, inputs: Tensor, labels: Tensor):
        # TODO: just get one batch

        outputs = self.explainer.classifier(inputs)
        loss_classification = cfg.LOSS(outputs, labels)
        loss = self.explainer._add_explanation_loss(self.critic_loader, loss_classification, 0)

        # self.explainer.train(train_loader,critic_loader)
        return loss

