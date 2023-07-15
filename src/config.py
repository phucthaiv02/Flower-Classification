import os
root = os.path.dirname(os.path.abspath(__file__))

IMAGE_SHAPE = (129, 129, 3)
N_CLASSES = 104

BATCH_SIZE = 64
EPOCHS = 20

TRAIN_DIR = os.path.join(os.path.dirname(root), 'data', 'train')
VAL_DIR = os.path.join(os.path.dirname(root), 'data', 'val')
TEST_DIR = os.path.join(os.path.dirname(root), 'data', 'test')
