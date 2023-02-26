import os
import random
import shutil
import cv2
from config import TEST_PATH, TRAIN_PATH, VALIDATE_PATH, X, Y

def move_images(src, dest, num_files):
    files = os.listdir(src)

    for file_name in random.sample(files, num_files):
        shutil.move(os.path.join(src, file_name), dest)

def create_black_and_white(data_path):
    src = "../" + data_path + Y
    files = os.listdir(src)
    for file_name in files:
        img = cv2.imread(src + file_name, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("../" + data_path + X + file_name, img)
        print("writing", file_name, "in", data_path)

def create_X_data():
    create_black_and_white(TRAIN_PATH)
    create_black_and_white(TEST_PATH)
    create_black_and_white(VALIDATE_PATH)