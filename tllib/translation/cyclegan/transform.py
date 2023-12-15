"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

import mindspore.nn as nn
from mindspore.dataset import transforms,vision


from tllib.vision.transforms import Denormalize


class Translation(nn.Cell):
    """
    Image Translation Transform Module

    Args:
        generator (torch.nn.Module): An image generator, e.g. :meth:`~tllib.translation.cyclegan.resnet_9_generator`
        device (torch.device): device to put the generator. Default: 'cpu'
        mean (tuple): the normalized mean for image
        std (tuple): the normalized std for image
    Input:
        - image (PIL.Image): raw image in shape H x W x C

    Output:
        raw image in shape H x W x 3

    """
    def __init__(self, generator, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super(Translation, self).__init__()
        self.generator = generator
        # self.device = device
        self.pre_process = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize(mean, std)
        ])
        self.post_process = transforms.Compose([
            Denormalize(mean, std),
            vision.ToPIL()
        ])

    def construct(self, image):
        image = self.pre_process(image.copy())  # C x H x W
        generated_image = self.generator(image.unsqueeze(dim = 0)).squeeze(dim = 0)
        return self.post_process(generated_image)
