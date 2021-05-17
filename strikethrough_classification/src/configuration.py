"""
Contains all code related to the configuration of experiments.
"""

import argparse
import random
import time
from configparser import SectionProxy, ConfigParser
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import torch


class ModelName(Enum):
    """
    Encodes the names of supported models.
    """
    DENSE = auto()
    RESNET = auto()

    @staticmethod
    def getByName(name: str) -> "ModelName":
        """
        Returns the ModelName corresponding to the given string. Returns ModelName.RESNET in case an unknown name is
        provided.

        Parameters
        ----------
        name : str
            string representation that should be converted to a ModelName

        Returns
        -------
            ModelName representation of the provided string, default: ModelName.RESNET
        """
        if name.upper() in [model.name for model in ModelName]:
            return ModelName[name.upper()]
        else:
            return ModelName.RESNET


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, fileSection: str = "DEFAULT"):
        self.fileSection = fileSection
        self.outDir = Path(parsedConfig.get('outdir')) / '{}_{}_{}'.format(fileSection, str(int(time.time())),
                                                                           random.randint(0, 100000))
        if not self.outDir.exists() and not test:
            self.outDir.mkdir(parents=True, exist_ok=True)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.epochs = parsedConfig.getint('epochs', 100)
        self.learningRate = parsedConfig.getfloat('learning_rate', 0.0002)
        self.betas = self.parseBetas(parsedConfig.get("betas", "0.5,0.999"))

        self.batchSize = parsedConfig.getint('batchsize', 4)
        self.imageHeight = parsedConfig.getint('imageheight', 128)
        self.imageWidth = parsedConfig.getint('imagewidth', 256)
        self.modelSaveEpoch = parsedConfig.getint('modelsaveepoch', 10)
        self.validationEpoch = parsedConfig.getint('validationEpochInterval', 10)
        self.trainImageDir = Path(parsedConfig.get('trainimgagebasedir'))
        self.testImageDir = Path(parsedConfig.get('testimagedir'))
        self.invertImages = parsedConfig.getboolean('invertImages', False)
        self.padScale = parsedConfig.getboolean('padscale', False)
        self.padWidth = parsedConfig.getint('padwidth', 512)
        self.padHeight = parsedConfig.getint('padheight', 256)

        self.modelName = ModelName.getByName(parsedConfig.get("model", "RESNET"))

        if not test:
            configOut = self.outDir / 'config.cfg'
            with configOut.open('w+') as cfile:
                parsedConfig.parser.write(cfile)

    @staticmethod
    def parseBetas(betaString: str) -> Tuple[float, float]:
        """
        Parses a comma-separated string to a list of floats.

        Parameters
        ----------
        betaString: str
            String to be parsed.

        Returns
        -------
            Tuple of floats.

        Raises
        ------
        ValueError
            if fewer than two values are specified
        """
        betas = betaString.split(',')
        if len(betas) < 2:
            raise ValueError("found fewer than two values for betas")
        return float(betas[0]), float(betas[1])


def getConfiguration() -> Configuration:
    """
    Reads the required arguments from command line and parse the respective configuration file/section.

    Returns
    -------
        parsed :class:`Configuration`
    """
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-config", required=False, help="section of config-file to use")
    cmdParser.add_argument("-configfile", required=False, help="path to config-file")
    args = vars(cmdParser.parse_args())
    fileSection = 'DEFAULT'
    fileName = 'config.cfg'
    if args["config"]:
        fileSection = args["config"]

    if args['configfile']:
        fileName = args['configfile']
    configParser = ConfigParser()
    configParser.read(fileName)
    parsedConfig = configParser[fileSection]
    sections = configParser.sections()
    for s in sections:
        if s != fileSection:
            configParser.remove_section(s)
    return Configuration(parsedConfig, fileSection=fileSection)
