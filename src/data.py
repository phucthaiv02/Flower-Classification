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


def load_data(data_dir, labeled=True):
    """
    Load data
    """
    if labeled:
        class_mode = 'categorical'
    else:
        class_mode = None

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.2,
        horizontal_flip=True
    )

    data = datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SHAPE[:2]),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return data


data = load_data(TRAIN_DIR)
show_image(data)
