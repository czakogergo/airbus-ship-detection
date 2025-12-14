import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import backend as K
from keras.losses import BinaryCrossentropy

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)          
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.cast(y_pred_f > 0.5, "float32")
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)




def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy()
    return bce(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

def f2_score_pixel(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = K.cast(y_pred_f > 0.5, "float32")

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    beta2 = 4.0  # F2 -> beta^2 = 4
    return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp + smooth)


def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def match_objects(gt_masks, pred_masks, iou_threshold):
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
    beta2 = beta ** 2
    numerator = (1 + beta2) * TP
    denominator = (1 + beta2) * TP + beta2 * FN + FP
    return numerator / denominator if denominator > 0 else 0.0


IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

def image_f2_score(gt_masks, pred_masks):
    scores = []

    for t in IOU_THRESHOLDS:
        TP, FP, FN = match_objects(gt_masks, pred_masks, t)
        scores.append(f2_score(TP, FP, FN))

    return np.mean(scores)

