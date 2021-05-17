"""
Utility module.
"""
from math import ceil

import torch
from PIL import Image, ImageOps
from torchvision import models
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose

from .configuration import Configuration, ModelName


class PadToSize:
    """
    Custom transformation that maintains the words original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height: int, width: int, padWith: int = 1):
        self.width = width
        self.height = height
        self.padWith = padWith

    def __call__(self, image: Image) -> Image:
        oldWidth, oldHeight = image.size
        if oldWidth != self.width or oldHeight != self.height:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = image.resize((intermediateWidth, self.height), resample=Image.BICUBIC)
            preprocessed = Image.new('L', (self.width, self.height), self.padWith)
            preprocessed.paste(resized)
            return preprocessed
        else:
            return image

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def composeTransformations(config: Configuration) -> Compose:
    """
    Composes various transformations based on the given experiment configuration. :class:`ToTensor` is always the final
    transformation.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Compose
        the composed transformations
    """
    transforms = []
    if config.padScale:
        transforms.append(PadToSize(config.padHeight, config.padWidth, 255))
    transforms.extend([Resize((config.imageHeight, config.imageWidth)), Grayscale(num_output_channels=1)])
    if config.invertImages:
        transforms.append(ImageOps.invert)
    transforms.append(ToTensor())
    return Compose(transforms)


def getModelByName(modelName: ModelName) -> torch.nn.Module:
    """
    Returns the model defined by modelName, adapted to single-channel inputs.

    Parameters
    ----------
    modelName : ModelName
        name of the model that shall be returned

    Returns
    -------
    torch.nn.Module
        Model definition. Default: ResNet18
    """
    if modelName == ModelName.DENSE:
        model = models.densenet121(progress=False, num_classes=2)
        # change densenet to single channel input:
        originalLayer = model.features.conv0
        model.features.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=originalLayer.out_channels,
                                               kernel_size=originalLayer.kernel_size,
                                               stride=originalLayer.stride, padding=originalLayer.padding,
                                               dilation=originalLayer.dilation, groups=originalLayer.groups,
                                               bias=originalLayer.bias, padding_mode=originalLayer.padding_mode)
    else:
        model = models.resnet18(progress=False, num_classes=2)
        originalLayer = model.conv1
        # change resnet to single channel input:
        model.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=originalLayer.out_channels,
                                      kernel_size=originalLayer.kernel_size,
                                      stride=originalLayer.stride, padding=originalLayer.padding,
                                      dilation=originalLayer.dilation, groups=originalLayer.groups,
                                      bias=originalLayer.bias, padding_mode=originalLayer.padding_mode)
    return model
