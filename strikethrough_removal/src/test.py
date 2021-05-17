"""
Code to test a previously trained neural network regarding its performance of removing strikethrough from a word.
"""
import argparse
import configparser
import logging
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from . import metrics
from .configuration import Configuration
from .dataset import TestDataset
from .utils import composeTransformations, getGeneratorModels

INFO_LOGGER_NAME = "st_removal"
RESULTS_LOGGER_NAME = "results"


def initLoggers(config: Configuration) -> None:
    """
    Utility function initialising a default info logger, as well as a results logger.

    Parameters
    ----------
    config : Configuration
        experiment configuration, to obtain the output location for file loggers

    Returns
    -------
        None
    """
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(config.outDir / "info.log", mode='w')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
    resultsLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(config.outDir / "{}.log".format(RESULTS_LOGGER_NAME), mode='w')
    fileHandler.setLevel(logging.INFO)
    resultsLogger.addHandler(fileHandler)


class TestRunner:
    """
    Utility class that wraps the initialisation and testing of a neural network.
    """

    def __init__(self, configuration: Configuration, saveCleanedImages: bool = True,
                 model_name: str = "genStrikeToClean_best_fmeasure.pth"):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.resultsLogger = logging.getLogger(RESULTS_LOGGER_NAME)
        self.config = configuration
        self.saveCleanedImages = saveCleanedImages

        transformations = composeTransformations(self.config)
        testDataset = TestDataset(self.config.testImageDir, transformations, strokeTypes=self.config.testStrokeTypes)
        self.validationDataloader = DataLoader(testDataset, batch_size=16, shuffle=False, num_workers=1)

        _, self.genStrikeToClean = getGeneratorModels(self.config)

        modelBasePath = self.config.outDir.parent
        state_dict = torch.load(modelBasePath / model_name, map_location=torch.device(self.config.device))
        if "model_state_dict" in state_dict.keys():
            state_dict = state_dict['model_state_dict']

        self.genStrikeToClean.load_state_dict(state_dict)
        self.genStrikeToClean.to(self.config.device)

        self.logger.info('Data dir: %s', str(self.config.testImageDir))
        self.logger.info('Validation dataset size: %d', len(testDataset))
        self.logger.info('Stroke types: %s', self.config.testStrokeTypes)

    def test(self) -> None:
        """
        Inititates the testing process.

        Returns
        -------
            None
        """
        self.genStrikeToClean.eval()
        self.resultsLogger.info('rmse,f1,strike_type,image_id')

        to_image = torchvision.transforms.ToPILImage()

        if self.saveCleanedImages:
            imgDir = self.config.outDir / "images"
            imgDir.mkdir(exist_ok=True, parents=True)
        else:
            imgDir = self.config.outDir

        with torch.no_grad():
            for datapoints in self.validationDataloader:
                strokeTypes = datapoints['strokeType']
                struckImages = datapoints['struck'].to(self.config.device)
                cleanedImages = self.genStrikeToClean(struckImages)
                groundTruthImages = datapoints['struckGt'].cpu().numpy()
                for idx, imagePath in enumerate(datapoints['path']):
                    strokeType = strokeTypes[idx]
                    cleanedImage = cleanedImages[idx]
                    groundTruth = groundTruthImages[idx].squeeze()

                    if self.saveCleanedImages:
                        img = to_image(cleanedImage)
                        img.save(imgDir / 'cleaned_{}'.format(imagePath))

                    cleanedImage = cleanedImage.cpu().numpy().squeeze()

                    rmse = metrics.calculateRmse(groundTruth, cleanedImage)[0]

                    if self.config.invertImages:
                        f1 = metrics.calculateF1Score(255.0 - groundTruth * 255.0,
                                                      255.0 - cleanedImage * 255.0, binarise=True)[0]
                    else:
                        f1 = metrics.calculateF1Score(groundTruth * 255.0, cleanedImage * 255.0,
                                                      binarise=True)[0]

                    self.resultsLogger.info('%f,%f,%s,%s', rmse, f1[0], strokeType, imagePath)


if __name__ == "__main__":
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-configfile", required=True, help="path to config file")
    cmdParser.add_argument("-data", required=True, help="path to data directory")
    cmdParser.add_argument("-save", required=False, help="saves cleaned images if given", default=False,
                           action='store_true')
    args = vars(cmdParser.parse_args())

    configPath = Path(args['configfile'])
    dataPath = Path(args['data'])

    configParser = configparser.ConfigParser()
    configParser.read(configPath)

    section = "DEFAULT"

    sections = configParser.sections()
    if len(sections) == 1:
        section = sections[0]
    else:
        logging.getLogger("st_recognition").warning(
            "Found %s than 1 section in config file. Using 'DEFAULT' as fallback.",
            'more' if len(sections) > 1 else 'fewer')

    parsedConfig = configParser[section]
    conf = Configuration(parsedConfig, test=True, fileSection=section)
    conf.testImageDir = dataPath

    out = configPath.parent / "{}_{}".format(dataPath.parent.name, dataPath.name)
    out.mkdir(exist_ok=True)
    conf.outDir = out

    initLoggers(conf)
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.info(conf.outDir)

    runner = TestRunner(conf, saveCleanedImages=args["save"])
    runner.test()
