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

def preprocess_imgs(set_name, img_size):
    """Resize and apply VGG-16 preprocessing.
    """
    set_new = list()
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))

    return np.array(set_new)

def augment_data(config, img_size):
    """Augment data based on preprocessed images.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        config['classification']['vgg16']['train_dir_crop'],
        color_mode='rgb',
        target_size=img_size,
        batch_size=32,
        class_mode='binary',
        seed=RANDOM_SEED)

    val_generator = test_datagen.flow_from_directory(
        config['classification']['vgg16']['val_dir_crop'],
        color_mode='rgb',
        target_size=img_size,
        batch_size=16,
        class_mode='binary',
        seed=RANDOM_SEED)

    return train_generator, val_generator

def plot_results(config, epochs_range, acc, val_acc, loss, val_loss):
    """Plot accuracy and loss of train and test set.
    """
    plt.figure(1)
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(acc, label='Train Set')
    plt.plot(val_acc, label='Val Set')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig(
        config['classification']['vgg16']['result_classification'] +
        'results_acc_loss.png')

def train_vgg16_model(config, train_generator, val_generator, img_size):
    """Training preweight vgg16 model.
    """
    vgg16_weight_path = config['classification']['vgg16']['vgg16_weights']

    base_model = VGG16(
        weights=vgg16_weight_path,
        include_top=False,
        input_shape=img_size + (3,))

    num_classes = 1

    model = Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    model.layers[0].trainable = False

    model.compile(
        loss = 'binary_crossentropy',
        metrics=['accuracy'],
        optimizer=RMSprop(lr=1e-4))

    model.summary()

    epochs = 3
    es = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=6)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=6,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=3,
        callbacks=[es])

    # Plot model performance
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(history.epoch) + 1)
    plot_results(config, epochs_range, acc, val_acc, loss, val_loss)

def main():
    """Main program
    """
    config = read_config_file(settings.config_file)
    if config['debug']:
        logging.basicConfig(level=logging.INFO)

    img_size = (224, 224)

    # Preprocess images, resize to apply vgg16
    # X_train_prep = preprocess_imgs(X_train_crop, img_size)
    # X_test_prep = preprocess_imgs(X_test_crop, img_size)
    # X_val_prep = preprocess_imgs(X_val_crop, img_size)
    # # plot_samples(X_train_prep, y_train, labels, 30)

    # Augment data for training
    train_generator, val_generator =  augment_data(config, img_size)

    # Train model
    train_vgg16_model(config, train_generator, val_generator, img_size)
    logging.info("Program was terminated!")

if __name__ == "__main__":
    main()


