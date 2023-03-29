import os
from config import TRAIN_PATH, X_PATH, Y_PATH, MODEL_FILE_NAME, DIM_ROWS, DIM_COLS, VALIDATE_PATH, MODEL_PATH
import tensorflow as tf
from tensorflow import keras
import numpy as np


def preprocess_data(src):
    X = np.zeros([0,DIM_ROWS,DIM_COLS,1])
    Y = np.zeros([0,DIM_ROWS,DIM_COLS,3])

    files = os.listdir(src + X_PATH)#[:100]
    for file_name in files:
        # convert to array - X
        imgX = keras.preprocessing.image.load_img(src + X_PATH + file_name, grayscale=True)
        imgX = np.expand_dims(keras.preprocessing.image.img_to_array(imgX), axis=0)
        X = np.append(X, imgX, axis=0)

        # convert to array - Y
        imgY = keras.preprocessing.image.load_img(src + Y_PATH + file_name)
        imgY = np.expand_dims(keras.preprocessing.image.img_to_array(imgY), axis=0)
        Y = np.append(Y, imgY, axis=0)

        print("Added", file_name)

    return X,Y

def train_model(X, Y, epochs=50, k=3, conv_layers=4):
    # keras model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(input_shape=(DIM_ROWS, DIM_COLS, 1), filters=64, kernel_size=(3, 3),
                                     padding="same", activation="relu"))

    for _ in range(conv_layers):
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(k, k), padding="same", activation="relu"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense((DIM_ROWS*DIM_COLS*3), activation="relu"))
    model.add(keras.layers.Reshape((DIM_ROWS,DIM_COLS,3)))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-7), loss='mean_squared_error')
    model.fit(X, Y, batch_size=50, epochs=epochs, verbose=1, validation_split=0)

    model.save(MODEL_PATH + MODEL_FILE_NAME)
    return model

def run_experiment(num_epochs, kernel_size, num_conv_layers):
    train_X, train_Y = preprocess_data(TRAIN_PATH)
    val_X, val_Y = preprocess_data(VALIDATE_PATH)
    model = train_model(train_X, train_Y, epochs=num_epochs, k=kernel_size, conv_layers=num_conv_layers)
    eval = model.evaluate(val_X, val_Y, batch_size=100, return_dict=False)
    return model

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    run_experiment(10,3,2)