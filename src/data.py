import os
from config import *

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def show_image(data):
    """
    Show image
    """
    batch = data.next()
    images = batch[0]
    labels = batch[1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))
    random = np.random.choice(range(len(images)), 6, replace=False)
    for i, idx in enumerate(random):
        axes[i//3, i % 3].imshow(images[idx])
        axes[i//3, i % 3].set_title(labels[idx])
        axes[i//3, i % 3].axis('off')

    plt.show()


def load_data():
    """
    Load data
    """

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_SHAPE[:2]),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_data = test_gen.flow_from_directory(
        VAL_DIR,
        target_size=(IMAGE_SHAPE[:2]),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_data, test_data

