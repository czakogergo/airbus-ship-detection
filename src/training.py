# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import backend as K
from keras import layers, Model
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


logger = setup_logger()

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(f)
    return f, p


def decoder_block(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet():
    inputs = layers.Input((config.IMG_SIZE, config.IMG_SIZE, 3))

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)

    return Model(inputs, outputs)


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


bce = BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
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


def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    train_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/train")
    val_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/val")
    test_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/test")

    model = build_unet()
    model.summary()
    model.compile(
        optimizer="adam",
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_score, f2_score_pixel]
    )

    checkpoint = ModelCheckpoint(
    config.MODEL_SAVE_PATH,
    monitor="val_loss",
    save_best_only=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    callbacks = [checkpoint, reduce_lr, early_stop]

    train_dataset = train_dataset.batch(32)
    val_dataset = val_dataset.batch(32)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()