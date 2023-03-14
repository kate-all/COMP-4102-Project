import numpy as np
import tensorflow as tf
from config import MODEL_PATH, TEST_PATH, X_PATH, Y_PATH, TEMP_PATH
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import cv2

# Environment/Backend stuff
import matplotlib as mpl
mpl.use('tkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def display_results(x, y, ground_truth=None):
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 1, 3

    if ground_truth is None:
        cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(x)
    plt.title("Original Image")

    fig.add_subplot(rows, cols, 2)
    plt.imshow(y)
    plt.title("Predicted")

    if ground_truth is not None:
        fig.add_subplot(rows, cols, 3)
        plt.imshow(ground_truth)
        plt.title("Ground Truth")

    plt.show()

def colourize(model, img_filename, show_ground_truth=True):
    # blur image
    x = cv2.imread(TEST_PATH + X_PATH + img_filename, cv2.IMREAD_GRAYSCALE)
    x = cv2.GaussianBlur(x, (7, 7), 0.5)
    cv2.imwrite(TEST_PATH + TEMP_PATH + img_filename, x)

    # convert to keras format
    x_img = keras.preprocessing.image.load_img(TEST_PATH + TEMP_PATH + img_filename)
    x = np.expand_dims(keras.preprocessing.image.img_to_array(x_img), axis=0)

    # predict
    y_hat = model.predict(x)
    y_hat = keras.preprocessing.image.array_to_img(y_hat[0])

    # display
    if show_ground_truth:
        y = Image.open(TEST_PATH + Y_PATH + img_filename)
        display_results(x_img, y_hat, ground_truth=y)
    else:
        display_results(x_img, y_hat)

if __name__ == "__main__":
    model = load_model()
    colourize(model, "0gfjPV.jpg")




