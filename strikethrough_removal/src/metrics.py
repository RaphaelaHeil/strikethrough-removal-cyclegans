"""
Metric related utility methods.
"""
import logging
from typing import List, Tuple

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import mean_squared_error
from torch import Tensor


def calculateRmse(actual_batch: np.ndarray, predicted_batch: np.ndarray) -> List[float]:
    """
    Calculates the RMSE for the given pairs of (actual,predicted) images

    Parameters
    ----------
    actual_batch : np.ndarray
        actual or expected image(s)
    predicted_batch : np.ndarray
        predicted image(s)

    Returns
    -------
    List[float]
        RMSE between each respective pair of (actual, predicted) images
    """
    if actual_batch.shape != predicted_batch.shape:
        raise ValueError('Inputs batches have different shapes!')

    rmse = []

    if len(actual_batch.shape) == 4:
        for idx in range(actual_batch.shape[0]):
            actual = actual_batch[idx].reshape(actual_batch.shape[2], actual_batch.shape[3])
            predicted = predicted_batch[idx].reshape(predicted_batch.shape[2], predicted_batch.shape[3])
            rmse.append(mean_squared_error(actual, predicted, squared=False))
    elif len(actual_batch.shape) == 2:
        rmse.append(mean_squared_error(actual_batch, predicted_batch, squared=False))
    elif len(actual_batch.shape) == 3:
        for idx in range(actual_batch.shape[0]):
            rmse.append(mean_squared_error(actual_batch[idx, :, :], predicted_batch[idx, :, :], squared=False))
    else:
        raise ValueError('Invalid shape input: ' + str(actual_batch.shape))
    return rmse


def _calculateF1ScorePerImage(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculates the F1 score, according to "Chaudhuri, B. B., & Adak, C. (2017). An approach for detecting and cleaning
    of struck-out handwritten text. Pattern Recognition, 61, 282-294." for ONE pair of (actual, predicted)

    Parameters
    ----------
    actual : np.ndarray
        actual (or expected) image
    predicted : nn.Tensor
        predicted image

    Returns
    -------
    Tuple[float, float, float]
        F1 score, detection rate and recognition accuracy per image
    """
    height, width = actual.shape
    pixelCount = width * height
    n = pixelCount - actual.sum() / 255.0
    m = pixelCount - predicted.sum() / 255.0
    s = actual + predicted
    zeros = np.zeros_like(s)
    zeros[s[:] == 0] = 1
    o2o = zeros.sum()
    if n == 0.0:
        detectionRate = 0.0
    else:
        detectionRate = o2o / n
    if m == 0.0:
        recognitionAccuracy = 0.0
    else:
        recognitionAccuracy = o2o / m
    if recognitionAccuracy + detectionRate == 0.0:
        return 0.0, 0.0, 0.0
    else:
        fmeasure = (2 * detectionRate * recognitionAccuracy) / (detectionRate + recognitionAccuracy)
    return fmeasure, detectionRate, recognitionAccuracy


def calculateF1Score(actual: np.ndarray, predicted: np.ndarray, binarise: bool = True) -> Tuple[
        List[float], List[float], List[float]]:
    """
    Calculates the F1 score, according to "Chaudhuri, B. B., & Adak, C. (2017). An approach for detecting and cleaning
    of struck-out handwritten text. Pattern Recognition, 61, 282-294." between pairs of (actual, predicted) images.

    Parameters
    ----------
    actual : np.ndarray
        actual (or expected) image(s)
    predicted : np.ndarray
        predicted image(s)
    binarise : bool
        if True, binarises the images before calculation, if False, will assume that they already are binarised

    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        F1 score, detection rate and recognition accuracy per image

    Raises
    ------
    ValueError
        if the shapes of actual and predicted don't match
    """
    if actual.shape != predicted.shape:
        raise ValueError('Inputs have different shapes!')
    if len(actual.shape) == 4:
        fmeasure = []
        detection_rate = []
        recognition_accuracy = []
        for idx in range(actual.shape[0]):
            a = actual[idx].reshape(actual.shape[2], actual.shape[3]).astype(np.uint8)
            p = predicted[idx].reshape(predicted.shape[2], predicted.shape[3]).astype(np.uint8)
            if binarise or len(np.unique(a)) > 2 or len(np.unique(p)) > 2:
                try:
                    a = (a > threshold_otsu(a)) * 255.0
                except ValueError:
                    logging.getLogger("st_removal").warning("Actual image only has one colour and therefore can't be "
                                                            "thresholded. Setting to all zeros for F1 calculation.")
                    a = np.zeros_like(a)
                try:
                    p = (p > threshold_otsu(p)) * 255.0
                except ValueError:
                    logging.getLogger("st_removal").warning("Predicted image only has one colour and therefore can't be"
                                                            " thresholded. Setting to all zeros for F1 calculation.")
                    p = np.zeros_like(p)
            results = _calculateF1ScorePerImage(a, p)
            fmeasure.append(results[0])
            detection_rate.append(results[1])
            recognition_accuracy.append(results[2])
        return fmeasure, detection_rate, recognition_accuracy
    else:
        if binarise:
            actual = (actual > threshold_otsu(actual)) * 255.0
            predicted = (predicted > threshold_otsu(predicted)) * 255.0
        results = _calculateF1ScorePerImage(actual, predicted)
        return [results[0]], [results[1]], [results[2]]
