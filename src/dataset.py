import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


def _augmentation():
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_datasets(
    data_dir=config.DATA_DIR,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    val_split=config.VAL_SPLIT,
    seed=config.SEED,
    cache=config.CACHE,
):
    data_dir = str(data_dir)
    if not tf.io.gfile.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}.\n"
            "Download TrashNet and place the class folders inside "
            "data/dataset-resized/ (see README)."
        )

    common = dict(
        directory=data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
    )

    train_ds = keras.utils.image_dataset_from_directory(
        validation_split=val_split, subset="training", **common
    )
    val_ds = keras.utils.image_dataset_from_directory(
        validation_split=val_split, subset="validation", **common
    )
    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    augment = _augmentation()

    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=autotune,
    )
    if cache:
        train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, class_names


def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, labels in dataset:
        counts += labels.numpy().sum(axis=0)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def add_sample_weights(dataset, class_weights):
    weights = tf.constant(class_weights, dtype=tf.float32)
    return dataset.map(
        lambda x, y: (x, y, tf.reduce_sum(y * weights, axis=-1)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def get_test_dataset(
    data_dir,
    img_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
):
    data_dir = str(data_dir)
    if not tf.io.gfile.isdir(data_dir):
        raise FileNotFoundError(f"Test dataset not found at {data_dir}.")

    test_ds = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    return test_ds.prefetch(tf.data.AUTOTUNE)
