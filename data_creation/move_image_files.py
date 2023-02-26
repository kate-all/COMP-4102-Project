import os
import random
import shutil

source = './data/train/ground_truth'
dest = './data/test/ground_truth'
files = os.listdir(source)
no_of_files = 1000 

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
