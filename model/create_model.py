import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_PATH, X_PATH, Y_PATH, MODEL_FILE_NAME, DIM_ROWS, DIM_COLS, TEMP, VALIDATE_PATH
from tensorflow import keras
import tensorflow as tf
import numpy as np

def preprocess_data(src):
    '''X = np.zeros([0,DIM_ROWS,DIM_COLS,1])
    Y = np.zeros([0,DIM_ROWS,DIM_COLS,3])'''

    files = os.listdir(src + X_PATH)
    X = keras.utils.image_dataset_from_directory(src + X_PATH, image_size=(DIM_ROWS, DIM_COLS), batch_size=20, labels=None, color_mode="grayscale")

    Y = keras.utils.image_dataset_from_directory(src + Y_PATH, image_size=(DIM_ROWS, DIM_COLS), batch_size=20, labels=None, color_mode="rgb")

    zipped_data = tf.data.Dataset.zip((X, Y))
    '''for file_name in files:
        # convert to array - X
        imgX = keras.preprocessing.image.load_img(src + X_PATH + file_name, grayscale=True)
        imgX = np.expand_dims(keras.preprocessing.image.img_to_array(imgX), axis=0)
        X = np.append(X, imgX, axis=0)

        # convert to array - Y
        imgY = keras.preprocessing.image.load_img(src + Y_PATH + file_name)
        imgY = np.expand_dims(keras.preprocessing.image.img_to_array(imgY), axis=0)
        Y = np.append(Y, imgY, axis=0)

        print("Added", file_name)
'''
    return zipped_data

def train_model(data, epochs=50, k=3, conv_layers=4):
    # keras model
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(input_shape=(DIM_ROWS, DIM_COLS, 1), filters=64, kernel_size=(k, k),
                                     padding="valid", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2)))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(k, k), padding="valid", activation="relu"))
    model.add(keras.layers.MaxPool2D((2, 2)))

    # upsampling
    model.add(keras.layers.UpSampling2D((2,2)))
    model.add(keras.layers.Conv2DTranspose(filters=32, kernel_size=(k,k), padding="valid", activation="relu"))

    model.add(keras.layers.UpSampling2D((2,2)))
    model.add(keras.layers.Conv2DTranspose(filters=64, kernel_size=(k,k), padding="valid", activation="relu"))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=(DIM_ROWS*DIM_COLS*3), activation='sigmoid' ))
    model.add(keras.layers.Reshape((DIM_ROWS,DIM_COLS,3)))

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-5), loss='mean_squared_error')
    model.fit(data, batch_size=20, epochs=epochs, verbose=2, validation_split=0)

    model.save(MODEL_FILE_NAME)
    return model

def run_experiment(num_epochs, kernel_size, num_conv_layers):
    train_data = preprocess_data(TRAIN_PATH)
    val_data = preprocess_data(VALIDATE_PATH)
    model = train_model(train_data, epochs=num_epochs, k=kernel_size, conv_layers=num_conv_layers)
    eval = model.evaluate(val_data, batch_size=100, return_dict=False)
    print("Validation Loss:", eval)

    return model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
run_experiment(500,5,4)
