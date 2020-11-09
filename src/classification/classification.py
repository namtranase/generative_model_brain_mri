import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

RANDOM_SEED = 123

IMG_PATH = 'data/kaggle_mri_classification/brain_tumor_dataset/'
SPLIT_PATH = 'data/kaggle_mri_classification/'
TRAIN_DIR = 'data/kaggle_mri_classification/TRAIN/'
TEST_DIR = 'data/kaggle_mri_classification/TEST/'
VAL_DIR = 'data/kaggle_mri_classification/VAL/'

def split_train_test_val():
    if not os.path.exists(SPLIT_PATH + 'TRAIN'):
            os.makedirs(SPLIT_PATH + 'TRAIN/YES')
            os.makedirs(SPLIT_PATH + 'TRAIN/NO')
            os.makedirs(SPLIT_PATH + 'TEST/YES')
            os.makedirs(SPLIT_PATH + 'TEST/NO')
            os.makedirs(SPLIT_PATH + 'VAL/YES')
            os.makedirs(SPLIT_PATH + 'VAL/NO')

    for CLASS in os.listdir(IMG_PATH):
        if CLASS.startswith('.'):
            return False
        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):
            img = IMG_PATH + CLASS + '/' + FILE_NAME
            if n < 5:
                des_test = SPLIT_PATH + 'TEST/' + CLASS.upper() + '/'
                shutil.copy(img, des_test)
            elif n < 0.8*IMG_NUM:
                des_train = SPLIT_PATH + 'TRAIN/' + CLASS.upper() + '/'
                shutil.copy(img, des_train)
            else:
                des_val = SPLIT_PATH + 'VAL/' + CLASS.upper() + '/'
                shutil.copy(img, des_val)

    return True

def main():
    # Train, test, val split
    status_split = split_train_test_val()
    print(status_split)

if __name__ == "__main__":
    main()


