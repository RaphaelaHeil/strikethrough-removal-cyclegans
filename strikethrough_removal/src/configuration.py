"""
Contains all code related to the configuration of experiments.
"""
import argparse
import configparser
import random
import time
from configparser import SectionProxy
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple

import torch


class StrikeThroughType(Enum):
    """
    Encodes the type of strikethrough
    """
    SINGLE_LINE = 0
    DOUBLE_LINE = 1
    DIAGONAL = 2
    CROSS = 3
    WAVE = 4
    ZIG_ZAG = 5
    SCRATCH = 6


class ExperimentType(Enum):
    """
    Encodes the type of experiment.
    """
    ORIGINAL = 0
    FEATURE_RECOG = 1
    STROKE_RECOG = 2
    NO_RECOG = 3

    @classmethod
    def getByName(cls, name: str) -> "ExperimentType":
        """
        Returns the ExperimentType corresponding to the given name.

        Parameters
        ----------
        name : str
            string that should be converted to a ExperimentType

        Returns
        -------
        ExperimentType
            ExperimentType representation of the provided string

        Raises
        ------
        ValueError
            if the given name does not correspond to a ExperimentType
        """
        name = name.upper()
        for t in ExperimentType:
            if t.name == name:
                return t
        raise ValueError("Unknown experiment type " + name)


class FeatureType(Enum):
    """
    Encodes the type of features to be appended to the input image.
    """
    NONE = 0
    SCALAR = 1
    TWO_CHANNEL = 2
    CHANNEL = 3
    CHANNEL_RANDOM = 4
    RANDOM = 5

    @classmethod
    def getByName(cls, name: str) -> "FeatureType":
        """
        Returns the FeatureType corresponding to the given name.

        Parameters
        ----------
        name : str
            string that should be converted to a FeatureType

        Returns
        -------
        FeatureType
            FeatureType representation of the provided string

        Raises
        ------
        ValueError
            if the given name does not correspond to a FeatureType
        """
        name = name.upper()
        for t in FeatureType:
            if t.name == name:
                return t
        raise ValueError("Unknown feature type " + name)


class ModelName(Enum):
    """
    Encodes the names of supported base models.
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
        if not test:
            self.outDir = Path(parsedConfig.get('outdir')) / '{}_{}_{}'.format(fileSection, str(int(time.time())),
                                                                               random.randint(0, 100000))
            parsedConfig['outdir'] = str(self.outDir)

        if not test and not self.outDir.exists():
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
        self.validationEpoch = parsedConfig.getint('validation', 10)
        self.trainImageDir = Path(parsedConfig.get('trainimgagebasedir'))
        self.testImageDir = Path(parsedConfig.get('testimagedir'))
        self.invertImages = parsedConfig.getboolean('invert_images', False)

        self.blockCount = parsedConfig.getint('blockcount', 6)

        self.poolSize = parsedConfig.getint('poolsize', 50)

        trainTypes = parsedConfig.get('train_stroke_types', 'all')
        self.trainStrokeTypes = self.parseStrokeTypes(trainTypes)

        testTypes = parsedConfig.get('test_stroke_types', '')
        self.testStrokeTypes = self.parseStrokeTypes(testTypes)
        if len(self.testStrokeTypes) < 1:
            self.testStrokeTypes = self.trainStrokeTypes

        self.count = parsedConfig.getint('count', 1000000)
        self.validationCount = parsedConfig.getint('val_count', 1000000)
        self.experiment = ExperimentType.getByName(parsedConfig.get('experiment', "original"))
        self.featureType = FeatureType.getByName(parsedConfig.get("featureType", "none"))
        if self.featureType != FeatureType.NONE and self.experiment == ExperimentType.ORIGINAL:
            self.featureType = FeatureType.NONE
            parsedConfig['featureType'] = "NONE"

        self.padScale = parsedConfig.getboolean('padscale', False)
        self.padWidth = parsedConfig.getint('padwidth', 512)
        self.padHeight = parsedConfig.getint('padheight', 128)
        self.cnnLambda = parsedConfig.getfloat('cnn_lambda', 0.5)
        self.identityLambda = parsedConfig.getfloat('identity_lambda', 0.5)
        self.cleanLambda = parsedConfig.getfloat('clean_lambda', 10.0)
        self.struckLambda = parsedConfig.getfloat('struck_lambda', 10.0)
        self.discWithFeature = parsedConfig.getboolean('disc_feature', False)
        if self.featureType == FeatureType.NONE:
            self.discWithFeature = False

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

    @staticmethod
    def parseStrokeTypes(strokeString: str) -> List[StrikeThroughType]:
        """
        Parses a comma-separated string to a list of stroke types.

        Parameters
        ----------
        strokeString : str
            string to be parsed

        Returns
        -------
        List[StrikeThroughType]
            list of stroke type strings

        """
        if '|' in strokeString:
            splitTypes = strokeString.split('|')  # for backward compatibility
        else:
            splitTypes = strokeString.split(',')
        strokeTypes = []
        if "all" in splitTypes:
            strokeTypes = ["all"]
        else:
            for item in splitTypes:
                item = item.strip()
                if item in [stroke.name for stroke in StrikeThroughType]:
                    strokeTypes.append(StrikeThroughType[item])
        if len(strokeTypes) < 1:
            strokeTypes = ["all"]
        return strokeTypes


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
    configParser = configparser.ConfigParser()
    configParser.read(fileName)
    parsedConfig = configParser[fileSection]
    sections = configParser.sections()
    for s in sections:
        if s != fileSection:
            configParser.remove_section(s)
    return Configuration(parsedConfig, fileSection=fileSection)
