import cv2
import os
from config import TRAIN_PATH, X_PATH, Y_PATH, MODEL_FILE_NAME, DIM_X, DIM_Y
import tensorflow as tf
from tensorflow import keras
import numpy as np

def preprocess_data(src):
    X = np.zeros([0,DIM_X,DIM_Y])
    Y = np.zeros([0,DIM_X,DIM_Y,3])

    files = os.listdir(src + X_PATH)[:5]
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

    return X,Y

def train_model(X, Y):
    # keras model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(input_shape=(DIM_X, DIM_Y, 1), filters=64, kernel_size=(3, 3),
                                     padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1200, activation='relu'))
    model.add(keras.layers.Dense(units=(DIM_X*DIM_Y*3), activation='sigmoid' ))
    model.add(keras.layers.Reshape((DIM_X,DIM_Y,3)))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
    model.fit(X, Y, batch_size=22, epochs=1, verbose=2, validation_split=0)

    model.save(MODEL_FILE_NAME)
    return model

if __name__ == "__main__":
    train_X, train_Y = preprocess_data("../" + TRAIN_PATH)
    model = train_model(train_X, train_Y)
    print()