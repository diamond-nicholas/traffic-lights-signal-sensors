import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
os.chdir('/Users/diamondnicholas/Desktop/Train')


data = []
labels = []
# We have 43 Classes
classes = 43
cur_path = os.getcwd()

print(cur_path)

# processing images
for i in range(classes):
    path = os.path.join(cur_path,'.',str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)

print(len(data))