"""
Utility module.
"""
import types
from math import ceil
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import ImageOps, Image
from torchvision import models
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor

from .configuration import ModelName, FeatureType, ExperimentType, Configuration
from .network import discriminator, generators


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
    transformation = []
    if config.padScale:
        transformation.append(PadToSize(config.padHeight, config.padWidth, 255))
    transformation.extend(
        [Resize((config.imageHeight, config.imageWidth)), Grayscale(num_output_channels=1)])
    if config.invertImages:
        transformation.append(ImageOps.invert)
    transformation.append(ToTensor())
    return Compose(transformation)


def __getInCount__(featureType: FeatureType) -> int:
    """
    Determines the number of input channels based on the given FeatureType.

    Parameters
    ----------
    featureType : FeatureType
        the type of feature that will be appended to the image input

    Returns
    -------
    int
        number of input channels
    """
    if featureType == FeatureType.NONE:
        in_count = 1
    elif featureType in (FeatureType.SCALAR, FeatureType.RANDOM):
        in_count = 2
    elif featureType == FeatureType.TWO_CHANNEL:
        in_count = 3
    else:
        in_count = 8
    return in_count


def getDiscriminatorModels(config: Configuration) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Prepares the CycleGAN discriminator models based on the given configuration.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module]
        clean discriminator, struck discriminator
    """

    in_count = __getInCount__(config.featureType)
    cleanDiscriminator = discriminator.NLayerDiscriminator(input_nc=1)
    if config.discWithFeature:
        struckDiscriminator = discriminator.NLayerDiscriminator(input_nc=in_count)
    else:
        struckDiscriminator = discriminator.NLayerDiscriminator(input_nc=1)
    return cleanDiscriminator, struckDiscriminator


def getGeneratorModels(config: Configuration) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Prepares the CycleGAN generator models based on the given configuration.

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Tuple[torch.nn.Module, torch.nn.Module]
        clean to struck generator, struck to clean generator
    """
    in_count = __getInCount__(config.featureType)
    if config.modelName == ModelName.DENSE:
        genCleanToStruck = generators.DenseGenerator(in_count, 1, norm_layer=torch.nn.BatchNorm2d,
                                                     n_blocks=config.blockCount)

        genStruckToClean = generators.DenseGenerator(1, 1, norm_layer=torch.nn.BatchNorm2d, n_blocks=config.blockCount)
    else:
        genCleanToStruck = generators.ResnetGenerator(in_count, 1, norm_layer=torch.nn.InstanceNorm2d,
                                                      n_blocks=config.blockCount)

        genStruckToClean = generators.ResnetGenerator(1, 1, norm_layer=torch.nn.InstanceNorm2d,
                                                      n_blocks=config.blockCount)
    return genCleanToStruck, genStruckToClean


def getPretrainedAuxiliaryLossModel(config: Configuration) -> Optional[torch.nn.Module]:
    """
    Initialises the pretrained model used for the auxiliary loss based on the given configuration

    Parameters
    ----------
    config : Configuration
        experiment configuration

    Returns
    -------
    Optional[torch.nn.Module]
        model (in eval mode), initialised from a pre-trained checkpoint or None, if ExperimentType is ORIGINAL or
        NO_RECOG

    """

    if config.experiment == ExperimentType.ORIGINAL or config.experiment == ExperimentType.NO_RECOG:
        return None

    baseDir = Path("checkpoints/rec")
    if config.padScale:
        if config.invertImages:
            path = baseDir / "pad_inv.pth"
        else:
            path = baseDir / "pad_noinv.pth"
    else:
        if config.invertImages:
            path = baseDir / "nopad_inv.pth"
        else:
            path = baseDir / "nopad_noinv.pth"

    model = models.densenet121(progress=False, num_classes=7)
    # change densenet to single channel input:
    originalLayer = model.features.conv0
    model.features.conv0 = torch.nn.Conv2d(in_channels=1, out_channels=originalLayer.out_channels,
                                           kernel_size=originalLayer.kernel_size,
                                           stride=originalLayer.stride, padding=originalLayer.padding,
                                           dilation=originalLayer.dilation, groups=originalLayer.groups,
                                           bias=originalLayer.bias, padding_mode=originalLayer.padding_mode)

    state_dict = torch.load(path, map_location=torch.device(config.device))

    if 'model_state_dict' in state_dict.keys():
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)

    if config.experiment == ExperimentType.FEATURE_RECOG:
        # drop the classification layer and return the flattened features
        def forward_replacement(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            return out

        model.forward = types.MethodType(forward_replacement, model)

    model = model.to(config.device)
    model.eval()
    return model
