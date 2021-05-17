"""
Code to train a CycleGAN to remove strikethrough from a struck-through word.
"""
import itertools
import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from . import configuration, dataset, metrics
from .configuration import ExperimentType, FeatureType, getConfiguration
from .network import image_pool
from .network.initialise import init_weights
from .utils import composeTransformations, getGeneratorModels, getDiscriminatorModels, getPretrainedAuxiliaryLossModel

INFO_LOGGER_NAME = "st_removal"
CLEAN_DISC_LOGGER_NAME = "cdLoss"
STRUCK_DISC_LOGGER_NAME = "sdLoss"
C_TO_S_GEN_LOGGER_NAME = "ctosLoss"
S_TO_C_GEN_LOGGER_NAME = "stocLoss"
VALIDATION_LOGGER_NAME = "validation"


def initLoggers(config: configuration.Configuration) -> None:
    """
    Utility function initialising a default info logger, as well as several loss loggers.

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

    fileHandler = logging.FileHandler(config.outDir / "info.log")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    for network in [CLEAN_DISC_LOGGER_NAME, STRUCK_DISC_LOGGER_NAME, C_TO_S_GEN_LOGGER_NAME, S_TO_C_GEN_LOGGER_NAME,
                    VALIDATION_LOGGER_NAME]:
        networkLogger = logging.getLogger(network)
        networkLogger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(config.outDir / "{}.log".format(network))
        fileHandler.setLevel(logging.INFO)
        networkLogger.addHandler(fileHandler)


class TrainRunner:
    """
    Utility class that wraps the initialisation, training and validation steps of the training run.
    """

    def __init__(self, config: configuration.Configuration):
        self.logger = logging.getLogger(INFO_LOGGER_NAME)
        self.ctosLogger = logging.getLogger(C_TO_S_GEN_LOGGER_NAME)
        self.stocLogger = logging.getLogger(S_TO_C_GEN_LOGGER_NAME)
        self.cdLogger = logging.getLogger(CLEAN_DISC_LOGGER_NAME)
        self.sdLogger = logging.getLogger(STRUCK_DISC_LOGGER_NAME)
        self.valLogger = logging.getLogger(VALIDATION_LOGGER_NAME)

        self.config = config

        transformations = composeTransformations(self.config)

        trainDataset = dataset.StruckCleanDataset(self.config.trainImageDir, transforms=transformations,
                                                  strokeTypes=self.config.trainStrokeTypes,
                                                  count=self.config.count, featureType=self.config.featureType)
        validationDataset = dataset.ValidationStruckCleanDataset(self.config.testImageDir,
                                                                 transforms=transformations,
                                                                 strokeTypes=self.config.testStrokeTypes,
                                                                 count=self.config.validationCount,
                                                                 featureType=self.config.featureType)

        self.trainDataLoader = DataLoader(trainDataset, batch_size=self.config.batchSize, shuffle=True, num_workers=1)
        self.validationDataloader = DataLoader(validationDataset, batch_size=self.config.batchSize, shuffle=False,
                                               num_workers=1)

        self.logger.info('Experiment: %s', self.config.experiment.name)
        self.logger.info('Data dir: %s', str(self.config.trainImageDir))
        self.logger.info('Train dataset size: %d', len(trainDataset))
        self.logger.info('Validation dataset size: %d', len(validationDataset))
        self.logger.info('Train stroke types: %s', self.config.trainStrokeTypes)
        self.logger.info('Val stroke types: %s', self.config.testStrokeTypes)

        self.genCleanToStruck, self.genStruckToClean = getGeneratorModels(self.config)

        self.genCleanToStruck.to(self.config.device)
        init_weights(self.genCleanToStruck)
        self.genStruckToClean.to(self.config.device)
        init_weights(self.genStruckToClean)

        self.cleanDiscriminator, self.struckDiscriminator = getDiscriminatorModels(self.config)
        self.struckDiscriminator.to(self.config.device)
        init_weights(self.struckDiscriminator)
        self.cleanDiscriminator.to(self.config.device)
        init_weights(self.cleanDiscriminator)

        self.generatorOptimiser = torch.optim.Adam(
            itertools.chain(self.genCleanToStruck.parameters(), self.genStruckToClean.parameters()),
            lr=self.config.learningRate, betas=self.config.betas)

        self.discriminatorOptimiser = torch.optim.Adam(
            itertools.chain(self.struckDiscriminator.parameters(), self.cleanDiscriminator.parameters()),
            lr=self.config.learningRate, betas=self.config.betas)

        self.discriminator_criterion = nn.MSELoss()
        self.image_l1_criterion = nn.L1Loss()

        self.fake_clean_pool = image_pool.ImagePool(self.config.poolSize)
        self.fake_struck_pool = image_pool.ImagePool(self.config.poolSize)

        if self.config.experiment in [ExperimentType.STROKE_RECOG, ExperimentType.FEATURE_RECOG]:
            self.strokeRecogniser = getPretrainedAuxiliaryLossModel(self.config)

        if self.config.experiment == ExperimentType.FEATURE_RECOG:
            self.cnn_loss_criterion = nn.L1Loss()
        elif self.config.experiment == ExperimentType.STROKE_RECOG:
            self.cnn_loss_criterion = nn.CrossEntropyLoss()

        self.bestRmse = float('inf')
        self.bestRmseEpoch = 0
        self.bestFscore = float('-inf')
        self.bestFscoreEpoch = 0

    def train(self) -> None:
        """
        Initiates the training process.

        Returns
        -------
            None
        """
        self.logger.info(self.config.device)
        self.valLogger.info("rmse, fmeasure")

        self.logger.info('-- Started training --')

        for epoch in range(1, self.config.epochs + 1):
            self.trainOneEpoch(epoch)
            if self.config.validationEpoch > 0 and epoch % self.config.validationEpoch == 0:
                self.validateOneEpoch(epoch)
        self.logger.info("best rmse: %f (%d), best fmeasure: %f (%d)", self.bestRmse, self.bestRmseEpoch,
                         self.bestFscore, self.bestFscoreEpoch)

    def validateOneEpoch(self, epoch: int) -> None:
        """
        Validates the neural network at the current training stage.

        Parameters
        ----------
        epoch : int
            current epoch number

        Returns
        -------
            None
        """
        rmses = []
        fmeasures = []

        epochdir = self.config.outDir / str(epoch)
        epochdir.mkdir(exist_ok=True, parents=True)

        self.genCleanToStruck.eval()
        self.genStruckToClean.eval()
        self.struckDiscriminator.eval()
        self.cleanDiscriminator.eval()
        with torch.no_grad():
            for batch_id, datapoints in enumerate(self.validationDataloader):
                cleanToStruckPairs = torch.Tensor().to(self.config.device)
                struckToCleanPairs = torch.Tensor().to(self.config.device)

                if self.config.featureType == FeatureType.NONE:
                    clean = datapoints["clean"]
                    struck = datapoints["struck"]
                    struckImageGroundTruth = datapoints["struckGt"]
                else:
                    clean = datapoints["clean"]
                    struck = datapoints["struck"]
                    struckImageGroundTruth = datapoints["struckGt"]
                    strokeFeature = datapoints["strokeFeature"]
                    strokeFeature = strokeFeature.to(self.config.device)

                clean = clean.to(self.config.device)
                struck = struck.to(self.config.device)
                struckImageGroundTruth = struckImageGroundTruth.to(self.config.device)

                # forward first part:
                if self.config.featureType == FeatureType.NONE:
                    generatedStruck = self.genCleanToStruck(clean)
                else:
                    generatedStruck = self.genCleanToStruck(torch.cat((clean, strokeFeature), dim=1))

                # forward second part:
                generatedClean = self.genStruckToClean(struck)

                tmp_cleanToStruck = torch.cat((clean, generatedStruck), 0)
                cleanToStruckPairs = torch.cat((cleanToStruckPairs, tmp_cleanToStruck), 0).to(self.config.device)
                tmp_struckToClean = torch.cat((struckImageGroundTruth, struck, generatedClean), 0)
                struckToCleanPairs = torch.cat((struckToCleanPairs, tmp_struckToClean), 0).to(self.config.device)

                save_image(cleanToStruckPairs, epochdir / "struckConcat_e{}_b{}.png".format(epoch, batch_id),
                           nrow=self.config.batchSize)

                save_image(struckToCleanPairs, epochdir / "cleanConcat_e{}_b{}.png".format(epoch, batch_id),
                           nrow=self.config.batchSize)

                cleanedImage = generatedClean.cpu().numpy()
                struckImageGroundTruth = struckImageGroundTruth.cpu().numpy()

                rmses.extend(metrics.calculateRmse(struckImageGroundTruth, cleanedImage))

                if self.config.invertImages:
                    fmeasures.extend(
                        metrics.calculateF1Score(255.0 - struckImageGroundTruth * 255.0,
                                                 255.0 - cleanedImage * 255.0, binarise=True)[0])
                else:
                    fmeasures.extend(
                        metrics.calculateF1Score(struckImageGroundTruth * 255.0, cleanedImage * 255.0,
                                                 binarise=True)[0])
        meanRMSE = np.mean(rmses)
        meanF = np.mean(fmeasures)
        self.logger.info('val [%d/%d], rmse: %f, fmeasure: %f', epoch, self.config.epochs, meanRMSE, meanF)

        self.valLogger.info("%f,%f", meanRMSE, meanF)

        if meanRMSE < self.bestRmse:
            self.bestRmseEpoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': self.genStruckToClean.state_dict(), },
                       self.config.outDir / Path('genStrikeToClean_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.genCleanToStruck.state_dict(), },
                       self.config.outDir / Path('cleanToStrike_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.struckDiscriminator.state_dict(), },
                       self.config.outDir / Path('struckDiscriminator_best_rmse.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.cleanDiscriminator.state_dict(), },
                       self.config.outDir / Path('cleanDiscriminator_best_rmse.pth'))
            self.logger.info('%d: Updated best rmse model', epoch)
            self.bestRmse = meanRMSE

        if meanF > self.bestFscore:
            self.bestFscoreEpoch = epoch
            torch.save({'epoch': epoch, 'model_state_dict': self.genStruckToClean.state_dict(), },
                       self.config.outDir / Path('genStrikeToClean_best_fmeasure.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.genCleanToStruck.state_dict(), },
                       self.config.outDir / Path('cleanToStrike_best_fmeasure.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.struckDiscriminator.state_dict(), },
                       self.config.outDir / Path('struckDiscriminator_best_fmeasure.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': self.cleanDiscriminator.state_dict(), },
                       self.config.outDir / Path('cleanDiscriminator_best_fmeasure.pth'))
            self.logger.info('%d: Updated best fmeasure model', epoch)
            self.bestFscore = meanF

    def trainOneEpoch(self, epoch: int) -> None:
        """
        Trains the neural network for one epoch.

        Parameters
        ----------
        epoch : int
            current epoch number

        Returns
        -------
            None
        """
        self.genCleanToStruck.train()
        self.genStruckToClean.train()
        self.struckDiscriminator.train()
        self.cleanDiscriminator.train()
        epochStartTime = time.time()
        totalDiscriminatorLossClean = []
        totalDiscriminatorLossStruck = []
        totalCleanToStruckGeneratorLoss = []
        totalStruckToCleanGeneratorLoss = []
        for batch_id, datapoints in enumerate(self.trainDataLoader):
            genStruckToCleanCycleLoss, genCleanToStruckCycleLoss, discriminatorLossClean, discriminatorLossStruck = (
                self.trainOneBatch(batch_id, datapoints, epoch))
            totalDiscriminatorLossClean.append(discriminatorLossClean)
            totalDiscriminatorLossStruck.append(discriminatorLossStruck)
            totalStruckToCleanGeneratorLoss.append(genStruckToCleanCycleLoss)
            totalCleanToStruckGeneratorLoss.append(genCleanToStruckCycleLoss)

        run_time = time.time() - epochStartTime
        self.logger.info('epoch [%d/%d], discriminator losses: clean %f, struck %f,'
                         ' generator losses: ctos %f, stoc %f, time:%f', epoch, self.config.epochs,
                         np.mean(totalDiscriminatorLossClean), np.mean(totalDiscriminatorLossStruck),
                         np.mean(totalCleanToStruckGeneratorLoss), np.mean(totalStruckToCleanGeneratorLoss), run_time)

        if epoch > 1 and self.config.modelSaveEpoch > 0 and epoch % self.config.modelSaveEpoch == 0:
            torch.save(self.genCleanToStruck.state_dict(),
                       self.config.outDir / Path('genCleanToStrike_epoch_{}.pth'.format(epoch)))
            torch.save(self.genStruckToClean.state_dict(),
                       self.config.outDir / Path('genStrikeToClean_epoch_{}.pth'.format(epoch)))
            torch.save(self.cleanDiscriminator.state_dict(),
                       self.config.outDir / Path('discriminatorClean_epoch_{}.pth'.format(epoch)))
            torch.save(self.struckDiscriminator.state_dict(),
                       self.config.outDir / Path('discriminatorStruck_epoch_{}.pth'.format(epoch)))

    def trainOneBatch(self, batchID: int, datapoints: Dict[str, Any], epoch: int) -> Tuple[float, float, float, float]:
        """
        Trains the neural network on a single batch.

        Parameters
        ----------
        batchID : int
            current batch number
        datapoints : Dict[str, Any]
            current batch datapoints
        epoch : int
            current epoch number

        Returns
        -------
        Tuple[float, float, float, float]
            genStruckToCleanCycleLoss, genCleanToStruckCycleLoss, discriminatorLossClean, discriminatorLossStruck
        """
        self.generatorOptimiser.zero_grad()

        if self.config.featureType == FeatureType.NONE:
            clean = datapoints["clean"]
            struck = datapoints["struck"]
            strokeType = datapoints["strokeType"]
            strokeFeature = None
        else:
            clean = datapoints["clean"]
            struck = datapoints["struck"]
            strokeType = datapoints["strokeType"]
            strokeFeature = datapoints["strokeFeature"]
            strokeFeature = strokeFeature.to(self.config.device)

        clean = clean.to(self.config.device)
        struck = struck.to(self.config.device)
        strokeType = strokeType.to(self.config.device)

        (generatedClean, generatedStruck, genStruckToCleanCycleLoss, genCleanToStruckCycleLoss) = self.trainGenerators(
            clean, struck, strokeFeature, strokeType)

        self.stocLogger.info("%d,%d,%f", epoch, batchID, genStruckToCleanCycleLoss)
        self.ctosLogger.info("%d,%d,%f", epoch, batchID, genCleanToStruckCycleLoss)

        discriminatorLossStruck, discriminatorLossClean = self.trainDiscriminators(generatedClean, clean,
                                                                                   generatedStruck, struck,
                                                                                   strokeFeature)

        self.sdLogger.info("%d,%d,%f", epoch, batchID, discriminatorLossStruck)
        self.cdLogger.info("%d,%d,%f", epoch, batchID, discriminatorLossClean)

        return genStruckToCleanCycleLoss, genCleanToStruckCycleLoss, discriminatorLossClean, discriminatorLossStruck

    def trainGenerators(self, clean: torch.Tensor, struck: torch.Tensor, strokeFeature: torch.Tensor,
                        strokeType: List[int]) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Trains the CycleGAN generators for one batch.

        Parameters
        ----------
        clean : torch.Tensor
            clean image(s)
        struck : torch.Tensor
            struck image(s)
        strokeFeature : torch.Tensor
            stroke feature representation, if applicable
        strokeType : List[int]
            list of stroke types, encoded as integers

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, float, float]
                generated clean word image, generated struck word image, genStruckToCleanCycleLoss,
                genCleanToStruckCycleLoss
        """
        # forward first part:
        if self.config.featureType == FeatureType.NONE:
            generatedStruck = self.genCleanToStruck(clean)
        else:
            generatedStruck = self.genCleanToStruck(torch.cat((clean, strokeFeature), dim=1))

        # forward second part:
        generatedClean = self.genStruckToClean(struck)

        # resconstruction:
        cycledClean = self.genStruckToClean(generatedStruck)
        if self.config.featureType == FeatureType.NONE:
            cycledStruck = self.genCleanToStruck(generatedClean)
        else:
            cycledStruck = self.genCleanToStruck(torch.cat((generatedClean, strokeFeature), dim=1))

        # identity:
        if self.config.identityLambda > 0.0:
            cleanIdentity = self.genStruckToClean(clean)
            if self.config.featureType == FeatureType.NONE:
                struckIdentity = self.genCleanToStruck(struck)
            else:
                struckIdentity = self.genCleanToStruck(torch.cat((struck, strokeFeature), dim=1))
            cleanIdentityLoss = self.image_l1_criterion(cleanIdentity,
                                                        clean) * self.config.identityLambda * self.config.struckLambda
            struckIdentityLoss = self.image_l1_criterion(struckIdentity,
                                                         struck) * self.config.identityLambda * self.config.cleanLambda
        else:
            cleanIdentityLoss = 0.0
            struckIdentityLoss = 0.0

        self.setDiscriminatorsRequiresGrad(False)

        if self.config.discWithFeature:
            struckDiscrimination = self.struckDiscriminator(torch.cat((generatedStruck, strokeFeature), dim=1))
        else:
            struckDiscrimination = self.struckDiscriminator(generatedStruck)

        cleanDiscrimination = self.cleanDiscriminator(generatedClean)

        lossStruckDiscriminator = self.discriminator_criterion(struckDiscrimination,
                                                               torch.ones_like(struckDiscrimination).to(
                                                                   self.config.device))
        lossCleanDiscriminator = self.discriminator_criterion(cleanDiscrimination,
                                                              torch.ones_like(cleanDiscrimination).to(
                                                                  self.config.device))

        genStruckToCleanCycleLoss = self.image_l1_criterion(cycledClean, clean) * self.config.cleanLambda

        genCleanToStruckCycleLoss = self.image_l1_criterion(cycledStruck, struck) * self.config.struckLambda

        if self.config.experiment == ExperimentType.STROKE_RECOG:
            predicted = self.strokeRecogniser(generatedStruck)
            predicted = torch.nn.functional.softmax(predicted, dim=1)
            recogniserLoss = self.cnn_loss_criterion(predicted, strokeType) * self.config.cnnLambda
            totalGeneratorLoss = (genStruckToCleanCycleLoss + genCleanToStruckCycleLoss + recogniserLoss +
                                  struckIdentityLoss + cleanIdentityLoss + lossStruckDiscriminator +
                                  lossCleanDiscriminator)
        elif self.config.experiment == ExperimentType.FEATURE_RECOG:
            realFeatures = self.strokeRecogniser(struck)
            fakeFeatures = self.strokeRecogniser(generatedStruck)
            recogniserLoss = self.cnn_loss_criterion(fakeFeatures, realFeatures) * self.config.cnnLambda
            totalGeneratorLoss = (genStruckToCleanCycleLoss + genCleanToStruckCycleLoss + recogniserLoss +
                                  struckIdentityLoss + cleanIdentityLoss + lossStruckDiscriminator +
                                  lossCleanDiscriminator)
        else:
            totalGeneratorLoss = (genCleanToStruckCycleLoss + genStruckToCleanCycleLoss + struckIdentityLoss +
                                  cleanIdentityLoss + lossStruckDiscriminator + lossCleanDiscriminator)

        totalGeneratorLoss.backward()
        self.generatorOptimiser.step()

        return (generatedClean, generatedStruck, genStruckToCleanCycleLoss.item(), genCleanToStruckCycleLoss.item())

    def trainDiscriminators(self, generatedClean: torch.Tensor, clean: torch.Tensor, generatedStruck: torch.Tensor,
                            struck: torch.Tensor, strokeFeature: torch.Tensor) -> Tuple[float, float]:
        """
        Trains the discriminators for one batch.

        Parameters
        ----------
        generatedClean : torch.Tensor
            generated clean image(s)
        clean : torch.Tensor
            original clean input image(s)
        generatedStruck : torch.Tensor
            generated struck-through image(s)
        struck : torch.Tensor
            original struck input image(s)
        strokeFeature : torch.Tensor
            stroke feature representation, if applicable

        Returns
        -------
        Tuple[float, float]
            struck discriminator loss, clean discriminator loss
        """
        self.setDiscriminatorsRequiresGrad(True)
        self.discriminatorOptimiser.zero_grad()

        if self.config.discWithFeature:
            fakeStruck = self.fake_struck_pool.query(torch.cat((generatedStruck, strokeFeature), dim=1).detach())
            realStruckPrediction = self.struckDiscriminator(torch.cat((struck, strokeFeature), dim=1))
            fakeStruckPrediction = self.struckDiscriminator(fakeStruck)
        else:
            fakeStruck = self.fake_struck_pool.query(generatedStruck.detach())
            realStruckPrediction = self.struckDiscriminator(struck)
            fakeStruckPrediction = self.struckDiscriminator(fakeStruck)

        fakeStruckLoss = self.discriminator_criterion(fakeStruckPrediction,
                                                      torch.zeros_like(fakeStruckPrediction).to(self.config.device))
        realStruckLoss = self.discriminator_criterion(realStruckPrediction,
                                                      torch.ones_like(realStruckPrediction).to(self.config.device))
        discriminatorLossStruck = (fakeStruckLoss + realStruckLoss) * 0.5
        discriminatorLossStruck.backward()

        fakeClean = self.fake_clean_pool.query(generatedClean.detach())

        realCleanPrediction = self.cleanDiscriminator(clean)
        fakeCleanPrediction = self.cleanDiscriminator(fakeClean)

        fakeCleanLoss = self.discriminator_criterion(fakeCleanPrediction,
                                                     torch.zeros_like(fakeCleanPrediction).to(self.config.device))
        realCleanLoss = self.discriminator_criterion(realCleanPrediction,
                                                     torch.ones_like(realCleanPrediction).to(self.config.device))
        discriminatorLossClean = (fakeCleanLoss + realCleanLoss) * 0.5
        discriminatorLossClean.backward()

        self.discriminatorOptimiser.step()

        return discriminatorLossStruck.item(), discriminatorLossClean.item()

    def setDiscriminatorsRequiresGrad(self, requiresGrad: bool) -> None:
        """
        Switches 'requires_grad' flag both discriminators according to :param:`requiresGrad`

        Parameters
        ----------
        requiresGrad : bool
            value to be propagated to 'requires_grad' of both discriminators

        Returns
        -------
        None
        """
        self.struckDiscriminator.requires_grad = requiresGrad
        self.cleanDiscriminator.requires_grad = requiresGrad


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    conf = getConfiguration()
    initLoggers(conf)
    logger = logging.getLogger(INFO_LOGGER_NAME)
    logger.info(conf.fileSection)
    runner = TrainRunner(conf)
    runner.train()
