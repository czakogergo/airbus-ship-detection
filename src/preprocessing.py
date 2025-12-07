import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import setup_logger
import config



logger = setup_logger()

def rle_decode(rle, shape=(768, 768)):
    """Masz képet generál az adott maszk adat alapján"""
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def preprocess_image_mask(image_path, mask_rles):
    """
    Betöltés, normalizálás és átméretezés
    """
    # Image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])

    # Mask
    mask = np.zeros((768, 768), dtype=np.float32)
    for rle in mask_rles:
        if pd.notna(rle):
            mask += rle_decode(rle)
    mask = np.clip(mask, 0, 1)
    mask = tf.image.resize(mask[..., None], [config.IMG_SIZE, config.IMG_SIZE], method='nearest')

    return img, mask


def create_dataset(image_dir, mask_csv):
    """
    Generál egy Tensorflow.Dataset típusú adatot az adott képekkel, és hozzájuk tartozó maszképekkel
    """
    mask_df = pd.read_csv(mask_csv)
    grouped = mask_df.groupby("ImageId")

    # Képek elérési útjai
    image_files = sorted(os.listdir(image_dir))[:100]
    image_paths = [os.path.join(image_dir, f) for f in image_files]

    # RLE listák
    rle_lists = [grouped.get_group(f)["EncodedPixels"].values if f in grouped.groups else [] for f in image_files]

    images = []
    masks = []
    counts = []
    for index in range(len(image_paths)):
        image, mask = preprocess_image_mask(image_paths[index], rle_lists[index])
        images.append(image)
        masks.append(mask)
        if str(rle_lists[index][0])=="nan":
            counts.append(0)
        else:
            counts.append(len(rle_lists[index]))

    dataset = tf.data.Dataset.from_tensor_slices((images, masks, counts))
    return dataset

def train_valid_test_datasetCreating():
    """
    Felosztja a tanító, validáló, és teszthalmazra.
    """
    dataset = tf.data.Dataset.load(config.IMAGES_MASK_DIR)

    train_dataset, test_dataset = tf.keras.utils.split_dataset(dataset,left_size=0.8, shuffle=True)
    train_dataset, val_dataset = tf.keras.utils.split_dataset(train_dataset,left_size=0.8, shuffle=True)

    train_dataset_notEmpty = train_dataset.filter(lambda img, mask, count: count > 0)
    train_dataset_empty = train_dataset.filter(lambda img, mask, count: count == 0)

    notEmpty_len = tf.data.experimental.cardinality(train_dataset_notEmpty).numpy()
    train_dataset_empty = train_dataset_empty.take(notEmpty_len)

    train_dataset = train_dataset_empty.concatenate(train_dataset_notEmpty)

    train_counts = train_dataset.map(lambda img, mask, count: count)
    val_counts = val_dataset.map(lambda img, mask, count: count)
    test_counts = test_dataset.map(lambda img, mask, count: count)

    train_dataset = train_dataset.map(lambda img, mask, count: (img, mask))
    val_dataset = val_dataset.map(lambda img, mask, count: (img, mask))
    test_dataset = test_dataset.map(lambda img, mask, count: (img, mask))

    return train_dataset, val_dataset, test_dataset


def preprocess():
    logger.info("Preprocessing data...")
    dataset = create_dataset(image_dir = (config.DATA_DIR + "/train_v2"), mask_csv = (config.DATA_DIR +"/" + config.MASKS_FILENAME))
    train_dataset, val_dataset, test_dataset =  train_valid_test_datasetCreating(dataset)
    train_dataset.save(config.IMAGES_MASK_DIR+"/train")
    val_dataset.save(config.IMAGES_MASK_DIR+"/val")
    test_dataset.save(config.IMAGES_MASK_DIR+"/test")

    logger.info("Preprocessing finished!")

if __name__ == "__main__":
    preprocess()