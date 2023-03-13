import numpy as np
import tensorflow as tf
from model import create_model
from config import MODEL_PATH, VALIDATE_PATH
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib as mpl
mpl.use('tkAgg')

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def run_prediction(model, photo, ground_truth=None):

    fig, ax = plt.subplots()
    img = keras.preprocessing.image.array_to_img(photo)
    ax.imshow(img)

    if ground_truth is not None:

        img = keras.preprocessing.image.array_to_img(ground_truth)
        ax.imshow(img)

    output = model.predict(np.expand_dims(photo, axis=0))

    img = keras.preprocessing.image.array_to_img(output[0])
    ax.imshow(img)
    plt.show()

if __name__ == "__main__":
    model = load_model()
    x_val, y_val = create_model.preprocess_data(VALIDATE_PATH)
    run_prediction(model, x_val[0], ground_truth=y_val[0])



