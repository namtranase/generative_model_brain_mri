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
        if path.startswith('.'):
            continue
        labels[i] = path
        for file in os.listdir(dir_path + path):
            if not file.startswith('.'):
                img = cv2.imread(dir_path + path + '/' + file)
                X.append(img)
                y.append(i)
        i += 1

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    logging.debug("%s images loaded from %s directory.", len(X), dir_path)

    return X, y, labels

def plot_confusion_matrix(
        cm, classes, normalize=False,
        title='Confusion matrix', cmap=plt.cm.Blues):
    """Plot confusion matrix for images.
    """
    # TODO: do it after
    return None

def plot_samples(X, y, lables_dict, n=50):
    """Plot grid for n images from specificed set.
    """
    for index in range(len(lables_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)
        plt.figure(figsize=(15, 6))
        c = 1
        for img in imgs:
            plt.subplot(i, j, c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1

        plt.suptitle('Tumor: {}'.format(lables_dict[index]))
        plt.savefig('src/classification/plot_sample.png')
        # plt.show()

def main():
    """Main program
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.DEBUG)

    # train, test, val split
    status_split = split_train_test_val(config)
    logging.info("Status of split oeration: %s",status_split)

    # Build data to training
    img_size = (224, 224)
    X_train, y_train, labels = load_data(
        config['classification']['vgg16']['train_dir'],
        img_size)
    X_test, y_test, _ = load_data(
        config['classification']['vgg16']['test_dir'],
        img_size
    )
    X_val, y_val, _ = load_data(
        config['classification']['vgg16']['val_dir'],
        img_size
    )

    # Plot image samples
    plot_samples(X_train, y_train, labels, 30)

    logging.info("Program was terminated!")

if __name__ == "__main__":
    main()


