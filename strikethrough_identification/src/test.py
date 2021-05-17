"""
Code to test a previously trained neural network regarding its performance of identifying whether a word image is
struck-through or not.
"""
import argparse
import configparser
import json
import logging
from itertools import compress
from pathlib import Path

import torch
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from . import configuration
from .dataset import CleanStruckDataset
from .utils import composeTransformations, getModelByName


class TestRunner:
    """
    Utility class that wraps the initialisation and testing of a neural network.
    """

    def __init__(self, config: configuration.Configuration):
        self.config = config

        transformations = composeTransformations(self.config)

        testDataSet = CleanStruckDataset(self.config.testImageDir, transforms=transformations)
        self.testDataloader = DataLoader(testDataSet, batch_size=self.config.batchSize, shuffle=False, num_workers=1)
        self.model = getModelByName(self.config.modelName)

        stateDict = torch.load(self.config.outDir / "best_f1.pth", map_location=torch.device(self.config.device))
        if 'model_state_dict' in stateDict.keys():
            stateDict = stateDict['model_state_dict']

        self.model.load_state_dict(stateDict)
        self.model = self.model.to(self.config.device)

    def test(self) -> None:
        """
        Inititates the testing process.

        Returns
        -------
            None
        """
        self.model.eval()
        predictedLabels = []
        expectedLabels = []
        misclassified = []

        with torch.no_grad():
            for datapoints in self.testDataloader:
                image = datapoints["image"].to(self.config.device)
                expectedLabel = datapoints["label"].to(self.config.device)
                paths = datapoints["path"]
                predicted = self.model(image)
                predicted = torch.nn.functional.softmax(predicted, dim=1)

                expected = expectedLabel.cpu().numpy()
                predicted = torch.max(predicted, dim=1).indices.cpu().numpy()
                expectedLabels.extend(expected.tolist())
                predictedLabels.extend(predicted.tolist())
                indices = (predicted - expected) != 0
                selected = list(compress(paths, indices.tolist()))
                misclassified.extend(selected)

        f1Score = f1_score(expectedLabels, predictedLabels)
        accuracy = accuracy_score(expectedLabels, predictedLabels)
        confusionMatrix = confusion_matrix(expectedLabels, predictedLabels)

        results = {"f1": f1Score, "accuracy": accuracy, "confusionMatrix": confusionMatrix.tolist(),
                   "confusionLabels": ["STRUCK", "CLEAN"], "misclassified": misclassified}

        filename = self.config.outDir / "{}_{}_results.json".format(self.config.testImageDir.parent.name,
                                                                    self.config.testImageDir.name)

        with open(filename, 'w', encoding="utf-8") as outFile:
            json.dump(results, outFile, indent=4)


if __name__ == "__main__":
    cmdParser = argparse.ArgumentParser()
    cmdParser.add_argument("-configfile", required=True, help="path to config file")
    cmdParser.add_argument("-data", required=True, help="path to data directory")
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
        logging.getLogger("st_identification").warning("Found %s than 1 section in config file. Using 'DEFAULT' as "
                                                       "fallback.", 'more' if len(sections) > 1 else 'fewer')

    parsedConfig = configParser[section]
    conf = configuration.Configuration(parsedConfig, test=True, fileSection=section)
    conf.outDir = configPath.parent

    conf.testImageDir = dataPath

    runner = TestRunner(conf)
    runner.test()
