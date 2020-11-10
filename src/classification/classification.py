"""
Copyright 2020 ASE Laboratory.
@author namtran.ase

Implement python class for classify mri brain images.
"""
import os
import itertools
import logging
from tqdm import tqdm

import shutil
import imutils
import cv2
import numpy as np

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

import settings
from src.config.config import read_config_file

RANDOM_SEED = 123

def split_train_test_val(config):
    """Split dataset to train, test and validation set.
    """
    img_path = config['classification']['vgg16']['img_path']
    split_path = config['classification']['vgg16']['split_path']

    if not os.path.exists(split_path + 'train'):
            os.makedirs(split_path + 'train/yes')
            os.makedirs(split_path + 'train/no')
            os.makedirs(split_path + 'test/yes')
            os.makedirs(split_path + 'test/no')
            os.makedirs(split_path + 'val/yes')
            os.makedirs(split_path + 'val/no')

    for _class in os.listdir(img_path):
        if _class.startswith('.'):
            return False
        img_num = len(os.listdir(img_path + _class))
        for (n, FILE_NAME) in enumerate(os.listdir(img_path + _class)):
            img = img_path + _class + '/' + FILE_NAME

            if n < 5:
                des_test = split_path + 'test/' + _class + '/'
                shutil.copy(img, des_test)
            elif n < 0.8*img_num:
                des_train = split_path + 'train/' + _class + '/'
                shutil.copy(img, des_train)
            else:
                des_val = split_path + 'val/' + _class + '/'
                shutil.copy(img, des_val)

    return True

def load_data(dir_path, img_size=(100, 100)):
    """Load and resize images.
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1

    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')

    return X, y, labels

def plot_confusion_matrix(
    cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix for images.
    """

    return None

def main():
    """Main program
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.INFO)

    # train, test, val split
    status_split = split_train_test_val(config)
    logging.info("Status of split oeration: %s",status_split)

    # Build data to training
    train_dir = 'data/kaggle_mri_classification/train/'
    test_dir = 'data/kaggle_mri_classification/test/'
    val_dir = 'data/kaggle_mri_classification/val/'
    logging.info("Program was terminated!")

if __name__ == "__main__":
    main()


