# Data paths
TRAIN_PATH = "data/train/"
TEST_PATH = "data/test/"
VALIDATE_PATH = "data/validate/"

X_PATH = "greyscale/"
Y_PATH = "ground_truth/"
TEMP = "temp/"

MODEL_FILE_NAME = "model.h5"
MODEL_PATH = "model/"

DOWNSAMPLE_ITERS = 2

DIM_ROWS = 256 // (2 ** DOWNSAMPLE_ITERS)
DIM_COLS = 256 // (2 ** DOWNSAMPLE_ITERS)


