import tensorflow as tf

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Jaccard / IoU:
    IoU = intersection / union
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Dice Score: 2 * TP / (2 * TP + FP + FN)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    return tf.reduce_mean((2 * intersection + smooth) / (union + smooth))