import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import backend as K
from keras.losses import BinaryCrossentropy

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Compute the Dice (F1) coefficient between ground truth and predictions.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        Dice coefficient as a scalar tensor.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)          
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    """Compute intersection-over-union (IoU) score for binary predictions.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        smooth: Smoothing constant to avoid division by zero.

    Returns:
        IoU score as a scalar tensor.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.cast(y_pred_f > 0.5, "float32")
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)




def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy and (1 - Dice) loss.

    This combines pixel-wise BCE with a Dice-based term to encourage
    overlap between predictions and ground truth.
    """
    bce = BinaryCrossentropy()
    return bce(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

def f2_score_pixel(y_true, y_pred, smooth=1e-6):
    """Compute per-pixel F2 score (beta=2) between ground truth and prediction.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        smooth: Small constant to avoid division by zero.

    Returns:
        Scalar F2 score.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.cast(y_pred_f > 0.5, "float32")

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    beta2 = 4.0  # F2 -> beta^2 = 4
    return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + smooth)


def iou(mask_a, mask_b):
    """Compute IoU for two binary masks (numpy arrays).

    Args:
        mask_a (np.ndarray): Binary mask A.
        mask_b (np.ndarray): Binary mask B.

    Returns:
        float: IoU value in [0, 1].
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_objects(gt_masks, pred_masks, iou_threshold):
    """Match predicted objects to ground-truth objects using IoU threshold.

    Finds one-to-one matches between predicted and ground-truth masks based
    on maximizing IoU, then returns counts of true positives, false
    positives and false negatives.

    Args:
        gt_masks (list[np.ndarray]): List of ground-truth binary masks.
        pred_masks (list[np.ndarray]): List of predicted binary masks.
        iou_threshold (float): Minimum IoU to consider a match.

    Returns:
        tuple: (TP, FP, FN) counts.
    """
    matched_gt = set()
    matched_pred = set()

    for i, p in enumerate(pred_masks):
        best_iou = 0
        best_j = -1

        for j, g in enumerate(gt_masks):
            if j in matched_gt:
                continue
            score = iou(g, p)
            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou >= iou_threshold:
            matched_pred.add(i)
            matched_gt.add(best_j)

    TP = len(matched_pred)
    FP = len(pred_masks) - TP
    FN = len(gt_masks) - TP

    return TP, FP, FN

def f2_score(TP, FP, FN, beta=2):
    """Compute F-beta score from TP/FP/FN counts.

    Args:
        TP (int): True positives.
        FP (int): False positives.
        FN (int): False negatives.
        beta (float): Beta parameter (>0). Default is 2 for F2.

    Returns:
        float: F-beta score.
    """
    beta2 = beta ** 2
    numerator = (1 + beta2) * TP
    denominator = (1 + beta2) * TP + beta2 * FN + FP
    return numerator / denominator if denominator > 0 else 0.0


IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def image_f2_score(gt_masks, pred_masks):
    """Compute an average F2 score for a single image across IoU thresholds.

    Uses the COCO-like evaluation approach averaging object-level F2
    scores over multiple IoU thresholds.

    Args:
        gt_masks (list[np.ndarray]): Ground-truth object masks.
        pred_masks (list[np.ndarray]): Predicted object masks.

    Returns:
        float: Mean F2 score over predefined IoU thresholds.
    """
    scores = []

    for t in IOU_THRESHOLDS:
        TP, FP, FN = match_objects(gt_masks, pred_masks, t)
        scores.append(f2_score(TP, FP, FN))

    return np.mean(scores)

