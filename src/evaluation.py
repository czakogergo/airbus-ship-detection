import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config
# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger
from evalutaionTools import evaluate_dataset_competition_f2

logger = setup_logger()



def evaluate():
    logger.info("Evaluating model...")
    model = load_model(config.MODEL_SAVE_PATH)

    
    test_f2 = evaluate_dataset_competition_f2(model, test_gen, pred_threshold=0.5)



if __name__ == "__main__":
    evaluate()