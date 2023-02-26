import os
import random
import shutil

def move_images(src, dest, num_files):
    files = os.listdir(src)

    for file_name in random.sample(files, num_files):
        shutil.move(os.path.join(src, file_name), dest)

def create_black_and_white(src, dest):
    pass