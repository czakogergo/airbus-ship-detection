import numpy as np
import cv2

def iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def extract_objects(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
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


def evaluate_dataset_competition_f2(model, test_gen, pred_threshold=0.5):
    all_scores = []

    for batch_idx in range(len(test_gen)):
        X, y = test_gen[batch_idx]            # X: képek, y: GT maszkok
        preds = model.predict(X, verbose=0)   # valószínűségi maszkok

        # Binarizálás
        preds_bin = (preds > pred_threshold).astype(np.uint8)

        batch_size = X.shape[0]

        for i in range(batch_size):
            gt = y[i]
            pr = preds_bin[i]

            # ha (H, W, 1) alakú, lapítsuk 2D-re
            if gt.ndim == 3:
                gt = gt[..., 0]
            if pr.ndim == 3:
                pr = pr[..., 0]

            score = f2_score_per_image(gt > 0.5, pr)  # GT-t is binárissá tesszük
            all_scores.append(score)

    return float(np.mean(all_scores))
