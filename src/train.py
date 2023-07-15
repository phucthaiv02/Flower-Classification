from model import *
import tensorflow as tf
model = build_model_from_scratch()
print(tf.config.list_physical_devices('GPU'))

print(model.summary())

