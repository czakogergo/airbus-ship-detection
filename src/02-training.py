# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from metricsTools import iou_metric, dice_coef

logger = setup_logger()

def conv_block(x, filters):
    """A convolutional block with two 3x3 Conv2D+ReLU layers.

    Args:
        x: Input tensor.
        filters (int): Number of convolution filters.

    Returns:
        Tensor after two convolutional layers.
    """
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def encoder_block(x, filters):
    """Encoder block: a conv_block followed by max-pooling.

    Returns the feature map before pooling and the pooled output.
    """
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(f)
    return f, p


def decoder_block(x, skip, filters):
    """Decoder block: upsample, concatenate skip connection, and convolve.

    Args:
        x: Input tensor to upsample.
        skip: Skip connection tensor from encoder.
        filters (int): Number of filters for conv_block.

    Returns:
        Tensor after upsampling, concatenation and convolution.
    """
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet():
    """Build a standard U-Net model for binary segmentation.

    Returns:
        Keras `Model` instance with a sigmoid output for mask prediction.
    """
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





def train():
    """Train the U-Net model using datasets saved to disk.

    Loads train/val/test datasets from `config.IMAGES_MASK_DIR`, builds the
    model, compiles it and runs `model.fit` with callbacks.
    """
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    train_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/train")
    val_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/val")
    test_dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR+"/test")
    
    model = build_unet()
    model.summary()

    checkpoint = ModelCheckpoint(
        config.MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,   # gyorsabb reagálás
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,  # kicsit több mozgástér
        restore_best_weights=True,
        verbose=1
    )

    callbacks = [checkpoint, reduce_lr, early_stop]


    train_dataset = train_dataset.batch(config.BATCH_SIZE)
    val_dataset = val_dataset.batch(config.BATCH_SIZE)

    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[iou_metric, dice_coef]
    )

    model.fit(train_dataset, 
                validation_data=val_dataset,
                epochs=config.EPOCHS,
                callbacks=callbacks,
                verbose=1)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()