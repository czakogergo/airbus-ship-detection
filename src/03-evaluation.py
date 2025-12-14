import numpy as np
import tensorflow as tf
from keras.models import load_model
import config
# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger
from evalutaionTools import evaluate_dataset_competition_f2
from metricsTools import bce_dice_loss, dice_coef, iou_score, f2_score_pixel
logger = setup_logger()

def evaluate():
    """Load the trained model and evaluate it on the test dataset.

    Loads model with necessary custom objects, runs predictions on the test
    dataset and computes the competition-style mean F2 score.
    """
    logger.info("Evaluating model...")
    custom_objects = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coef": dice_coef,
    "iou_score": iou_score,
    "f2_score_pixel": f2_score_pixel,
    # If the model uses a Keras/TF built-in BCE, you don't need to pass 'bce'
    # but since bce_dice_loss calls bce defined above, it is internally available.
}
    model = load_model(
        config.MODEL_SAVE_PATH
        ,custom_objects=custom_objects)
    test_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/test")
    test_dataset = test_dataset.batch(2)
    test_f2 = evaluate_dataset_competition_f2(model, test_dataset, pred_threshold=0.5)
    logger.info(f"Test Score: {test_f2}")
    logger.info("Evaluating ended")

if __name__ == "__main__":
    evaluate()