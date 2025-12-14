import os
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import setup_logger
import config



logger = setup_logger()

def rle_decode(rle, shape=(768, 768)):
    """Decode a run-length encoded (RLE) mask into a 2D binary array.

    Args:
        rle (str): Run-length encoding string (space-separated starts and lengths).
        shape (tuple): Desired output mask shape as (height, width).

    Returns:
        np.ndarray: 2D array of dtype uint8 with values 0 or 1 representing the mask.
    """
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def preprocess_image_mask(image_path, mask_rles):
    """Load and preprocess an image and its associated masks.

    This function reads an image from disk, decodes and normalizes it,
    resizes it to the model input size, decodes any RLE masks, combines
    them into a single binary mask and resizes the mask to the model size.

    Args:
        image_path (str): Path to the image file.
        mask_rles (iterable): Iterable of RLE strings for the image masks.

    Returns:
        tuple: (image_tensor, mask_tensor) where image is float32 normalized
               and mask is a single-channel binary mask.
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
    """Create a TensorFlow Dataset from image files and an RLE mask CSV.

    Reads mask encodings from `mask_csv` and pairs them with images found
    in `image_dir`. Produces a dataset of (image, mask, count) tuples where
    `count` is the number of encoded masks for that image.

    Args:
        image_dir (str): Directory containing image files.
        mask_csv (str): CSV file path with columns ['ImageId', 'EncodedPixels'].

    Returns:
        tf.data.Dataset: Dataset yielding (image, mask, count) for each image.
    """
    mask_df = pd.read_csv(mask_csv)
    grouped = mask_df.groupby("ImageId")

    # Képek elérési útjai
    image_files = sorted(os.listdir(image_dir))[:config.DATA_SIZE]
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

def train_valid_test_datasetCreating(dataset):
    """Split a dataset into training, validation and test sets and balance positives.

    Splits the provided dataset into train/validation/test with an 80/20 split
    and then balances the training split so that the number of images
    without masks equals the number of images with masks.

    Args:
        dataset (tf.data.Dataset): Dataset yielding (image, mask, count) tuples.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) where each yields
               (image, mask) pairs.
    """
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
    """Main preprocessing entrypoint: builds datasets and saves them to disk.

    Reads images and masks, splits into train/val/test and saves the
    resulting datasets to `config.IMAGES_MASK_DIR`.
    """
    logger.info("Preprocessing data...")
    dataset = create_dataset(image_dir = (config.DATA_DIR + "/train_v2"), mask_csv = (config.DATA_DIR +"/" + config.MASKS_FILENAME))
    train_dataset, val_dataset, test_dataset =  train_valid_test_datasetCreating(dataset)
    train_dataset.save(config.IMAGES_MASK_DIR+"/train")
    val_dataset.save(config.IMAGES_MASK_DIR+"/val")
    test_dataset.save(config.IMAGES_MASK_DIR+"/test")

    logger.info("Preprocessing finished!")

if __name__ == "__main__":
    preprocess()