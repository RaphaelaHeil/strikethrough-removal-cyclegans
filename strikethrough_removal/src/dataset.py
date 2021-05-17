"""
Datasets for loading clean and struck-through word images.
"""
import logging
from pathlib import Path
from typing import List, Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .configuration import FeatureType, StrikeThroughType

CLEAN_DIR_NAME = "clean"
STRUCK_DIR_NAME = "struck"
STRUCK_GT_DIR_NAME = "struck_gt"


class StruckCleanDataset(Dataset):
    """
    Dataset containing unrelated clean and struck word images.
    """

    def __init__(self, rootDir: Path, transforms: Compose = None,
                 strokeTypes: List[Union[str, StrikeThroughType]] = None, count: int = 1000000,
                 featureType: FeatureType = FeatureType.NONE):
        Dataset.__init__(self)
        self.rootDir = Path(rootDir)
        self.cleanDir = rootDir / CLEAN_DIR_NAME
        self.struckDir = rootDir / STRUCK_DIR_NAME
        self.cleanFileNames = []
        self.struckFileNames = []
        self.fileStrokeType = []
        if not strokeTypes:
            self.strokeTypes = ["all"]
        else:
            self.strokeTypes = strokeTypes
        self.featureType = featureType
        for fileName in self.cleanDir.iterdir():
            if fileName.suffix in ['.png', '.jpg', '.jpeg']:
                self.cleanFileNames.append(fileName)

        csvGlob = list(self.struckDir.glob("*.csv"))
        if len(csvGlob) > 1:
            logger = logging.getLogger("str_poc")
            logger.warning("found more than one csv in train dir; using the first one")
        if len(csvGlob) < 1:
            raise FileNotFoundError("no csv file found in stroke directory")
        csvFile = csvGlob[0]
        struckDf = pd.read_csv(csvFile, dtype={'image_id': str, 'writer_id': str, 'strike_type': str})
        if "all" in self.strokeTypes:
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in struckDf.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                struckDf.iloc]
        else:
            hasCorrectType = struckDf["strike_type"].isin([stroke.name for stroke in self.strokeTypes])
            selectedRows = struckDf[hasCorrectType]
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in selectedRows.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                selectedRows.iloc]

        if count < len(self.struckFileNames):
            self.struckFileNames = self.struckFileNames[:count]
            self.cleanFileNames = self.cleanFileNames[:count]
            self.fileStrokeType = self.fileStrokeType[:count]

        self.count = len(self.struckFileNames)
        self.transforms = transforms

    def __len__(self) -> int:
        return self.count

    def __getStruckIndex(self, cleanIndex: int) -> int:
        # pylint: disable=unused-argument
        return np.random.randint(0, self.count)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the struck image and its stroke type based on the given index. Randomly draws an index for a clean
        image and prepares the associated strokeFeature (if any) based on the struck image's stroke type.
        Parameters
        ----------
        index : int
            index at which to retrieve the clean image

        Returns
        -------
        Dict[str, Any]
            dictionary with 'clean', 'struck' and 'strokeType', as well as 'strokeFeature' if applicable
        """
        cleanImage = Image.open(self.cleanFileNames[index]).convert('RGB')

        struckIndex = self.__getStruckIndex(index)
        struckImage = Image.open(self.struckFileNames[struckIndex]).convert('RGB')
        strokeType = self.fileStrokeType[struckIndex][1].value

        if self.transforms:
            cleanImage = self.transforms(cleanImage)
            struckImage = self.transforms(struckImage)

        if self.featureType == FeatureType.NONE:
            return {"clean": cleanImage, "struck": struckImage, "strokeType": strokeType}

        if self.featureType == FeatureType.SCALAR:
            strokeFeature = torch.ones_like(cleanImage) * (strokeType + 1 / 7.0)
            return {"clean": cleanImage, "struck": struckImage, "strokeFeature": strokeFeature,
                    "strokeType": strokeType}

        if self.featureType == FeatureType.RANDOM:
            strokeFeature = torch.ones_like(cleanImage) * torch.rand(cleanImage.shape)
            return {"clean": cleanImage, "struck": struckImage, "strokeFeature": strokeFeature,
                    "strokeType": strokeType}

        if self.featureType == FeatureType.TWO_CHANNEL:
            strokeFeature = torch.ones((2, cleanImage.shape[1], cleanImage.shape[2]))
            strokeFeature[0, :] = strokeFeature[0, :] * (strokeType + 1 / 7.0)
            strokeFeature[1, :] = strokeFeature[1, :] * torch.rand(cleanImage.shape)
            return {"clean": cleanImage, "struck": struckImage, "strokeFeature": strokeFeature,
                    "strokeType": strokeType}

        if self.featureType == FeatureType.CHANNEL:
            strokeFeature = torch.zeros((7, cleanImage.shape[1], cleanImage.shape[2]))
            strokeFeature[strokeType] = 1.0
            return {"clean": cleanImage, "struck": struckImage, "strokeFeature": strokeFeature,
                    "strokeType": strokeType}

        if self.featureType == FeatureType.CHANNEL_RANDOM:
            strokeFeature = torch.zeros((7, cleanImage.shape[1], cleanImage.shape[2]))
            strokeFeature[strokeType] = 1.0
            strokeFeature = strokeFeature * torch.rand(cleanImage.shape)
            return {"clean": cleanImage, "struck": struckImage, "strokeFeature": strokeFeature,
                    "strokeType": strokeType}

        return {"clean": cleanImage, "struck": struckImage, "strokeType": strokeType}


class ValidationStruckCleanDataset(StruckCleanDataset):
    """
    Extends :class:`StruckCleanDataset` with the ground truth image for each struck-through word
    """

    def __init__(self, rootDir: Path, transforms: Compose = None,
                 strokeTypes: List[Union[str, StrikeThroughType]] = None, count: int = 1000000,
                 featureType: FeatureType = FeatureType.NONE):
        StruckCleanDataset.__init__(self, rootDir, transforms, strokeTypes, count, featureType)

        self.struckGTDir = Path(rootDir) / STRUCK_GT_DIR_NAME

        self.count = len(self.struckFileNames)
        self.transforms = transforms

    def __len__(self) -> int:
        return self.count

    def __getStruckIndex(self, cleanIndex: int) -> int:
        return cleanIndex

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the struck image and its stroke type and ground truth based on the given index. Randomly draws an index
        for a clean image and prepares the associated strokeFeature (if any) based on the struck image's stroke type.
        Parameters
        ----------
        index : int
            index at which to retrieve the clean image

        Returns
        -------
        Dict[str, Any]
            dictionary with 'clean', 'struck', 'struckGt' and 'strokeType', as well as 'strokeFeature' if applicable
        """
        outDict = StruckCleanDataset.__getitem__(self, index)

        imageName = self.struckFileNames[index].name
        struckGTPath = self.struckGTDir / imageName

        struckImageGroundTruth = Image.open(struckGTPath).convert('RGB')

        if self.transforms:
            struckImageGroundTruth = self.transforms(struckImageGroundTruth)

        outDict["struckGt"] = struckImageGroundTruth

        return outDict


class TestDataset(Dataset):
    """
    Dataset containing unrelated struck word images and their related clean ground truth version.
    """

    def __init__(self, rootDir: Path, transforms: Compose = None,
                 strokeTypes: List[Union[str, StrikeThroughType]] = None):
        self.rootDir = Path(rootDir)
        self.struckDir = rootDir / STRUCK_DIR_NAME
        self.struckGTDir = rootDir / STRUCK_GT_DIR_NAME
        self.struckFileNames = []
        self.struckGTFileNames = []
        self.fileStrokeType = []
        if not strokeTypes:
            self.strokeTypes = ["all"]
        self.strokeTypes = strokeTypes

        csvGlob = list(self.struckDir.glob("*.csv"))
        if len(csvGlob) > 1:
            logger = logging.getLogger("st_removal")
            logger.warning("found more than one csv in train dir; using the first one")
        if len(csvGlob) < 1:
            raise FileNotFoundError("no csv file found in stroke directory")
        csvFile = csvGlob[0]
        struckDf = pd.read_csv(csvFile, dtype={'image_id': str, 'writer_id': str, 'strike_type': str})
        if "all" in self.strokeTypes:
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in struckDf.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                struckDf.iloc]
        else:
            hasCorrectType = struckDf["strike_type"].isin([stroke.name for stroke in self.strokeTypes])
            selectedRows = struckDf[hasCorrectType]
            self.struckFileNames = [(self.struckDir / row.image_id).with_suffix(".png") for row in selectedRows.iloc]
            self.fileStrokeType = [
                ((self.struckDir / row.image_id).with_suffix(".png"), StrikeThroughType[row.strike_type]) for row in
                selectedRows.iloc]

        for fileName in self.struckFileNames:
            name = fileName.name
            self.struckGTFileNames.append(self.struckGTDir / name)

        self.count = len(self.struckFileNames)
        self.transforms = transforms

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the struck image and its clean ground truth, as well as image name and stroke type, for the given index.

        Parameters
        ----------
        index : int
            index at which to retrieve the datapoint

        Returns
        -------
        Dict[str, Any]
            dictionary with 'struck', 'struckGt', 'path' and 'strokeType'
        """
        struckImage = Image.open(self.struckFileNames[index]).convert('RGB')
        struckImageGroundTruth = Image.open(self.struckGTFileNames[index]).convert('RGB')
        strokeType = self.fileStrokeType[index][1].name

        if self.transforms:
            struckImage = self.transforms(struckImage)
            struckImageGroundTruth = self.transforms(struckImageGroundTruth)
        return {'struck': struckImage, 'struckGt': struckImageGroundTruth,
                'path': str(self.struckFileNames[index].name), 'strokeType': strokeType}
