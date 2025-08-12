import psutil
import os

TRAIN_DATA_PATH = os.path.join("data/BraTS2021")
TEST_DATA_PATH = os.path.join("data/BraTS2017")
OUTPUT_PATH = os.path.join("./output/")
INFERENCE_FILENAME = "2d_unet_decathlon"

"""
If the batch size is too small, then training is unstable.
I believe this is because we are using 2D slicewise model.
There are more slices without tumor than with tumor in the
dataset so the data may become imbalanced if we choose too
small of a batch size. There are, of course, many ways
to handle imbalance training samples, but if we have
enough memory, it is easiest just to select a sufficiently
large batch size to make sure we have a few slices with
tumors in each batch.
"""
BATCH_SIZE = 128
EPOCHS = 2

LEARNING_RATE = 0.0001
WEIGHT_DICE_LOSS = 0.8

FEATURE_MAPS = 16
PRINT_MODEL = False

BLOCK_TIME = 0
NUM_INTER_THREADS = 1
# NUM_INTRA_THREADS = psutil.cpu_count(logical=False)
NUM_INTRA_THREADS = 3

CROP_DIM = 96
SEED = 122333
TRAIN_TEST_SPLIT = 0.80

CHANNELS_FIRST = False

USE_UP_SAMPLING = False
USE_AUGMENTATION = False

USE_DROPOUT = True
DROPOUT_RATE = 0.25
USE_P_CONV = False

FONT_FAMILY = "New SEC Keypad"
FONT_SIZE = 12
DPI = 300
