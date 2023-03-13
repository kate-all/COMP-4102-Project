import numpy as np
import tensorflow as tf
from model import create_model
from config import MODEL_PATH, VALIDATE_PATH
import matplotlib.pyplot as plt
from tensorflow import keras

# Environment/Backend stuff
import matplotlib as mpl
mpl.use('tkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def run_prediction(model, photo, ground_truth=None):

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 1, 3

    if ground_truth is None:
        cols = 2

    fig.add_subplot(rows, cols, 1)
    img = keras.preprocessing.image.array_to_img(photo)
    plt.imshow(img)
    plt.title("Original Image")

    output = model.predict(np.expand_dims(photo, axis=0))

    fig.add_subplot(rows, cols, 2)
    img = keras.preprocessing.image.array_to_img(output[0])
    plt.imshow(img)
    plt.title("Predicted")

    if ground_truth is not None:
        fig.add_subplot(rows, cols, 3)
        img = keras.preprocessing.image.array_to_img(ground_truth)
        plt.imshow(img)
        plt.title("Ground Truth")

    plt.show()

if __name__ == "__main__":
    model = load_model()
    x_val, y_val = create_model.preprocess_data(VALIDATE_PATH)
    run_prediction(model, x_val[0], ground_truth=y_val[0])



