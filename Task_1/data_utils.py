"""
data_utils.py
Shared CIFAR-10 loading + preprocessing for Task 1 (Part A and Part B).

IMPORTANT: This module fixes the random seed and the train/val/test split
so that Part A (traditional CNN) and Part B (customized CNN) are compared
on exactly the same data. Always import load_data() from here in both
scripts instead of loading CIFAR-10 separately.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

SEED = 42
VAL_FRACTION = 0.1  # 10% of the training set is held out for validation

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_global_seed(seed: int = SEED) -> None:
    """Fix seeds across numpy / tensorflow / python random for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    """
    Loads CIFAR-10, normalizes pixel values to [0, 1], and produces a fixed
    train/validation/test split.

    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
        y_* are one-hot encoded, shape (N, 10)
    """
    set_global_seed(SEED)

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Fixed, reproducible train/val split using a seeded permutation
    rng = np.random.RandomState(SEED)
    n_total = x_train_full.shape[0]
    indices = rng.permutation(n_total)

    n_val = int(n_total * VAL_FRACTION)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_val = to_categorical(y_val, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    print("Train:", x_train.shape, y_train.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)
