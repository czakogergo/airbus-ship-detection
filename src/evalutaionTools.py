import numpy as np
import tensorflow as tf
import cv2
from metricsTools import image_f2_score 

def iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def extract_objects(binary_mask):
    binary_mask = tf.cast(binary_mask, tf.uint8)
    binary_mask = binary_mask.numpy()
    if tf.is_tensor(binary_mask):
        binary_mask = binary_mask.numpy()

    # 2. Float → bináris
    binary_mask = (binary_mask > 0.5)

    # 3. uint8
    binary_mask = binary_mask.astype(np.uint8)

    # 4. Batch és channel eltávolítás
    binary_mask = np.squeeze(binary_mask)
    num_labels, labels = cv2.connectedComponents(binary_mask)

    objects = []
    for label in range(1, num_labels):  # 0 = háttér
        obj_mask = (labels == label)
        objects.append(obj_mask)

    return objects



def f2_score_per_image(gt_mask, pred_mask):
    thresholds = np.arange(0.5, 1.0, 0.05)

    gt_objects   = extract_objects(gt_mask)
    pred_objects = extract_objects(pred_mask)

    # Speciális esetek
    if len(gt_objects) == 0 and len(pred_objects) == 0:
        return 1.0   

    if len(gt_objects) == 0 and len(pred_objects) > 0:
        return 0.0   

    f2_scores = []

    for t in thresholds:
        TP = 0
        FP = 0

        matched_gt = set()

        for pred_obj in pred_objects:
            best_iou = 0.0
            best_gt_idx = None

            # keressük a legjobb GT match-et
            for i, gt_obj in enumerate(gt_objects):
                if i in matched_gt:
                    continue
                iou_val = iou(pred_obj, gt_obj)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = i

            if best_iou >= t and best_gt_idx is not None:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1

        FN = len(gt_objects) - len(matched_gt)

        beta2 = 4.0  # beta=2 => beta^2 = 4
        denom = (1 + beta2) * TP + beta2 * FN + FP
        if denom == 0:
            f2_t = 0.0
        else:
            f2_t = (1 + beta2) * TP / denom

        f2_scores.append(f2_t)

    return float(np.mean(f2_scores))


def evaluate_dataset_competition_f2(model, test_dataset, pred_threshold=0.5):
    all_scores = []
    for test_data in test_dataset:
        X, y = test_data            # X: képek, y: GT maszkok
        batch_size = len(test_data)
        for i in range(batch_size):
            preds = model.predict(X[i].numpy()[np.newaxis, ...], verbose=0)   # valószínűségi maszkok
        # Binarizálás
            preds_bin = (preds > pred_threshold).astype(np.uint8)
            
            score = f2_score_per_image(y[i] > 0.5, preds_bin)  # GT-t is binárissá tesszük
            all_scores.append(score)

    return float(np.mean(all_scores))


