"""
Contains the dataset for loading word images with different types of strikethrough
"""
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class StrikeThroughType(Enum):
    """
    Encodes the type of strikethrough in an image as one out of seven kinds.
    """
    SINGLE_LINE = 0
    DOUBLE_LINE = 1
    DIAGONAL = 2
    CROSS = 3
    WAVE = 4
    ZIG_ZAG = 5
    SCRATCH = 6

    @staticmethod
    def valueByName(name: str) -> int:
        """
        Returns the integer value assigned to the StrikeThroughType with the matching name.
        Parameters
        ----------
        name : str
            name of the strikethrough type

        Returns
        -------
        int:
            value assigned to the given name

        Raises
        ------
        ValueError
            if the given name does not match a :class:`StrikeThroughType`
        """
        if name in [strokeType.name for strokeType in StrikeThroughType]:
            return StrikeThroughType[name].value
        else:
            raise ValueError("Unknown name {}".format(name))


class StruckDataset(Dataset):
    """
    Dataset containing words with different types of strikethrough applied to them.
    """

    def __init__(self, rootDir: Path, transforms: Compose = None):
        Dataset.__init__(self)
        self.rootDir = Path(rootDir)
        self.transforms = transforms
        self.data = []
        csvGlob = list(self.rootDir.glob("*.csv"))
        if len(csvGlob) > 1:
            logger = logging.getLogger("st_recognition")
            logger.warning("found more than one csv in train dir; using the first one")
        if len(csvGlob) < 1:
            raise FileNotFoundError("no csv file found in stroke directory '{}'".format(self.rootDir))
        csvFile = csvGlob[0]
        struckDf = pd.read_csv(csvFile, dtype={'image_id': str, 'writer_id': str, 'strike_type': str})
        self.data = [
            ((self.rootDir / row.image_id).with_suffix(".png"), StrikeThroughType.valueByName(row.strike_type)) for
            row in struckDf.iloc]
        self.count = len(self.data)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the image and associated strikethrough type label and image path at the given index.

        Parameters
        ----------
        index : int
            index at which to retrieve the datapoint

        Returns
        -------
            dictionary with 'image', 'label' and 'path' for the given index
        """
        filename, strokeType = self.data[index]
        image = Image.open(filename).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return {"image": image, "label": strokeType, "path": str(filename.relative_to(filename.parents[1]))}
