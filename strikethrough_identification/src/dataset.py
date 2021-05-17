"""
Contains the dataset for loading clean vs struck images.
"""
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

CLEAN_DIR = "clean"
STRUCK_DIR = "struck"
STRUCK_GT_DIR = "struck_gt"
IMAGE_EXTENSION = "*.png"


class CleanStruckDataset(Dataset):
    """
    Dataset containing samples of clean and struck words.
    """

    def __init__(self, rootDir: Path, transforms: Compose = None, count: int = None):
        Dataset.__init__(self)
        self.rootDir = Path(rootDir)
        self.transforms = transforms
        self.data = []
        struckDir = self.rootDir / STRUCK_DIR
        struckFiles = list(struckDir.glob(IMAGE_EXTENSION))
        if count and count < len(struckFiles):
            np.random.shuffle(struckFiles)
            self.data.extend([(f, 0) for f in struckFiles[:count]])
        else:
            self.data.extend([(f, 0) for f in struckFiles])
        cleanDir = self.rootDir / CLEAN_DIR
        if not cleanDir.exists():
            cleanDir = self.rootDir / STRUCK_GT_DIR
            if not cleanDir.exists():
                raise FileNotFoundError(
                    "Neither '{}' nor '{}' exist as directories in base '{}'".format(CLEAN_DIR, STRUCK_GT_DIR, rootDir))

        cleanFiles = list(cleanDir.glob(IMAGE_EXTENSION))
        if count and count < len(cleanFiles):
            np.random.shuffle(cleanFiles)
            self.data.extend([(f, 1) for f in cleanFiles[:count]])
        else:
            self.data.extend([(f, 1) for f in cleanFiles])
        self.count = len(self.data)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns the image and associated clean/struck label and image path at the given index.

        Parameters
        ----------
        index : int
            index at which to retrieve the datapoint

        Returns
        -------
        Dict[str, Any]
            dictionary with 'image', 'label' and 'path' for the given index
        """
        filename, clean = self.data[index]
        image = Image.open(filename).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return {"image": image, "label": clean, "path": str(filename.relative_to(filename.parents[1]))}
