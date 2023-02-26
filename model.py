import cv2
import os
from config import TRAIN_PATH, X_PATH, Y_PATH
import tensorflow as tf
from tensorflow import keras
import numpy as np

def preprocess_data(src) -> (tf.Tensor, tf.Tensor):
    X = np.zeros([0,256,256])
    Y = np.zeros([0,256,256,3])
    files = os.listdir(src + X_PATH)#[:5]
    for file_name in files:
        imgX = cv2.imread(src + X_PATH + file_name, cv2.IMREAD_GRAYSCALE)

        # gaussian filter
        img_filtered = cv2.GaussianBlur(imgX, (7, 7), 0.5)

        # convert to array - X
        X_img_array = np.asarray([img_filtered])
        X = np.append(X, X_img_array, axis=0)

        # convert to array - Y
        imgY = cv2.imread(src + Y_PATH + file_name, cv2.IMREAD_COLOR)
        Y_img_array = np.asarray([imgY])
        Y = np.append(Y, Y_img_array, axis=0)

        print("Added", file_name)

    return tf.convert_to_tensor(X, dtype=float), tf.convert_to_tensor(Y, dtype=float)

def train_model(X, Y):
    # keras model
    model = keras.Sequential()

    return model

if __name__ == "__main__":
    train_X, train_Y = preprocess_data(TRAIN_PATH)
    model = train_model(train_X, train_Y)

    print("Test")
