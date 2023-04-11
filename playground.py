import numpy as np
from config import MODEL_PATH, TEST_PATH, X_PATH, Y_PATH, TEMP, MODEL_FILE_NAME, VALIDATE_PATH, TRAIN_PATH
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import cv2
from model.create_model import preprocess_data

# Environment/Backend stuff
import matplotlib as mpl
mpl.use('tkAgg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def display_results(x, y, ground_truth=None):
    fig = plt.figure(figsize=(10, 7))
    rows, cols = 1, 3

    if ground_truth is None:
        cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(x[:,:,::-1])
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
    x_read = cv2.imread(TEST_PATH + X_PATH + img_filename, cv2.IMREAD_GRAYSCALE)
    x_read = cv2.GaussianBlur(x_read, (7, 7), 0.5)
    cv2.imwrite(TEST_PATH + TEMP + img_filename, x_read)

    # convert to keras format
    x_img = keras.preprocessing.image.load_img(TEST_PATH + TEMP + img_filename, grayscale=True)
    x = np.expand_dims(keras.preprocessing.image.img_to_array(x_img), axis=0)

    # predict
    y_hat = model.predict(x)
    y_hat = keras.preprocessing.image.array_to_img(y_hat[0])

    # display
    x = cv2.cvtColor(x_read, cv2.COLOR_GRAY2RGB)
    if show_ground_truth:
        y = Image.open(TEST_PATH + Y_PATH + img_filename)
        display_results(x, y_hat, ground_truth=y)
    else:
        display_results(x, y_hat)

def evaluate(model):
    train_data = preprocess_data(TRAIN_PATH)
    val_data = preprocess_data(VALIDATE_PATH)
    test_data = preprocess_data(TEST_PATH)

    print("TRAIN LOSS:", model.evaluate(train_data, batch_size=100))
    print("VALIDATION LOSS:", model.evaluate(val_data, batch_size=100))
    print("TEST LOSS:", model.evaluate(test_data, batch_size=100))

model = keras.models.load_model(MODEL_PATH + MODEL_FILE_NAME)
# colourize(model, "lighter.jpg")
#colourize(model, "beach.jpg")
#colourize(model, "woman_with_hat.jpg")

# evaluate(model)




