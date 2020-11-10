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

    X = np.array(X)
    y = np.array(y)
    logging.info("%s images loaded from %s directory.", len(X), dir_path)

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
        plt.savefig(
            'src/classification/vgg16_model/plot_{}.png'.format(lables_dict[index]))

def crop_imgs(set_name, add_pixels_value=0):
    """Crop images to precise with input of vgg16, remove black coners.
    """
    set_new = list()
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        #Remove any small region of noise
        thresh = cv2.threshold(gray, 45, 255,cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        #Find contours
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        #Find the extreme points
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        add_pixels = add_pixels_value
        new_img = img[ext_top[1] - add_pixels:ext_bot[1] + add_pixels,
                      ext_left[0] - add_pixels:ext_right[0] + add_pixels].copy()
        set_new.append(new_img)

    return np.array(set_new)

def save_crop_images(x_set, y_set, file_dir):
    """Save crop images.
    """
    if not os.path.exists(file_dir):
            os.makedirs(file_dir + 'yes')
            os.makedirs(file_dir + 'no')

    i = 0
    for (img, img_class) in zip(x_set, y_set):
        if img_class == 0:
            cv2.imwrite(file_dir + 'no/' + str(i) + '.jpg', img)
        else:
            cv2.imwrite(file_dir + 'yes/' + str(i) + '.jpg', img)
        i += 1

def main():
    """Main program
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.INFO)

    # train, test, val split
    status_split = split_train_test_val(config)
    logging.info("Status of split operation: %s",status_split)

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
    # plot_samples(X_train, y_train, labels, 30)

    # Crop images
    X_train_crop = crop_imgs(set_name=X_train)
    X_test_crop = crop_imgs(set_name=X_test)
    X_val_crop = crop_imgs(set_name=X_val)

    plot_samples(X_train_crop, y_train, labels, 30)

    # Save croped images
    save_crop_images(
        X_train_crop,
        y_train,
        config['classification']['vgg16']['train_dir_crop'])
    save_crop_images(
        X_test_crop,
        y_test,
        config['classification']['vgg16']['test_dir_crop'])
    save_crop_images(
        X_val_crop,
        y_val,
        config['classification']['vgg16']['val_dir_crop'])

    logging.info("Program was terminated!")

if __name__ == "__main__":
    main()


