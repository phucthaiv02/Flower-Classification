
import tensorflow as tf

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D

from config import *


def build_model_from_scratch():
    """
    Train model from scratch
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
              input_shape=IMAGE_SHAPE, activation='relu', name='conv_1'))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool_1'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
              activation='relu', name='conv_3'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv_4'))
    model.add(MaxPooling2D(pool_size=(2, 3), name='pool_2'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
              activation='relu', name='conv_5'))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv_6'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool_3'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='dense_3'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, name='output'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    return model
