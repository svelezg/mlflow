#! /usr/bin/env python3
"""
The data set used is from https://www.kaggle.com/c/titanic/data
Modeling survival chances for Titanic passengers
"""

import os
import warnings
import sys

import mlflow
import keras
import pandas as pd
import numpy as np
import zipfile
import logging


def unzip_and_load(zipfolder, filename):
    """
    unzip folder and load raw dara
    :param zipfolder: zip folder path
    :param filename: file_ path
    :return: raw_data
        Raw data set
    """
    with zipfile.ZipFile(zipfolder, 'r') as zip_ref:
        zip_ref.extractall()

    data = pd.read_csv(filename)

    print('data set shape: ', data.shape, '\n')
    print(data.head())

    return data


def csv_from_url(csv_url):
    """
    loads data from csv url
    :param csv_url: example:
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    :return: data
    """
    try:
        data = pd.read_csv(csv_url, sep=";")
        print('data set shape: ', data.shape, '\n')
        print(data.head())
        return data
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, "
            "check your internet connection. Error: %s", e
        )


def clean_data(data, drop_cols, fill_cols):
    """
    Perform drop and fill
    :param data:
    :param drop_cols:
    :param fill_cols:
    :return:
    """
    # drop
    data = data.drop(labels=drop_cols, axis=1)

    # zero to nan
    data['Fare'] = data['Fare'].replace(0, np.nan)

    # fill Fare
    data['Fare'] = data.groupby(['Pclass', 'Sex'])['Fare'] \
        .transform(lambda x: x.fillna(x.mean()))

    # fill Age
    data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'] \
        .transform(lambda x: x.fillna(x.mean()))

    print('data set shape: ', data.shape, '\n')
    print(data.head())
    print(data.columns[data.isna().any()].tolist())

    return data


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    :param nx: number of input features to the network
    :param layers: list containing the number of nodes
        in each layer of the network
    :param activations: list containing the activation
        functions used for each layer of the network
    :param lambtha: L2 regularization parameter
    :param keep_prob: probability that a node will be kept for dropout
    :return: keras model
    """
    # input placeholder
    inputs = keras.Input(shape=(nx,))

    # regularization scheme
    reg = keras.regularizers.L1L2(l2=lambtha)

    # a layer instance is callable on a tensor, and returns a tensor.
    # first densely-connected layer
    my_layer = keras.layers.Dense(units=layers[0],
                                  activation=activations[0],
                                  kernel_regularizer=reg,
                                  input_shape=(nx,))(inputs)

    # subsequent densely-connected layers:
    for i in range(1, len(layers)):
        my_layer = keras.layers.Dropout(1 - keep_prob)(my_layer)
        my_layer = keras.layers.Dense(units=layers[i],
                                      activation=activations[i],
                                      kernel_regularizer=reg,
                                      )(my_layer)

    network = keras.Model(inputs=inputs, outputs=my_layer)

    return network


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model
    with categorical crossentropy loss and accuracy metrics
    :param network: model to optimize
    :param alpha: learning rate
    :param beta1: first Adam optimization parameter
    :param beta2: second Adam optimization parameter
    :return: None
    """
    network.compile(optimizer=keras.optimizers.Adam(alpha, beta1, beta2),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return None


def train_model(network, data, labels, batch_size, epochs,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False,
                validation_split=0.1):
    """
        trains a model using mini-batch gradient descent
        :param network: model to train
        :param data: numpy.ndarray of shape (m, nx) containing the input data
        :param labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        :param batch_size: size of the batch used for mini-batch grad descent
        :param epochs: number of passes through data for mini-batch grad desc
        :param validation_data:  data to validate the model with, if not None
        :param early_stopping: boolean that indicates whether
            early stopping should be used
        :param patience: patience used for early stopping
        :param learning_rate_decay: boolean that indicates whether
            learning rate decay should be used
        :param alpha: initial learning rate
        :param decay_rate: decay rate
        :param save_best: boolean indicating whether to save the model
            after each epoch if it is the best
        :param filepath: file path where the model should be saved
        :param verbose: boolean that determines if output should be printed
            during training
        :param shuffle: boolean that determines whether to shuffle the batches
            every epoch.
            Normally, it is a good idea to shuffle,
            but for reproducibility, we have chosen to set the default to False
        :return: History object generated after training the model
        """

    def learning_rate(epoch):
        """ updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    callback_list = []

    # models save callback
    if save_best:
        mcp_save = keras.callbacks.ModelCheckpoint(filepath,
                                                   save_best_only=True,
                                                   monitor='val_loss',
                                                   mode='min')
        callback_list.append(mcp_save)

    # learning rate decay callback
    if learning_rate_decay:
        lrd = keras.callbacks.LearningRateScheduler(learning_rate,
                                                    verbose=1)
        callback_list.append(lrd)

    # early stopping callback
    if early_stopping:
        es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=patience,
                                           restore_best_weights=True)
        callback_list.append(es)

    # training
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          callbacks=callback_list,
                          validation_split=validation_split)

    return history


if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.keras.autolog()

        logging.basicConfig(level=logging.WARN)
        logger = logging.getLogger(__name__)

        warnings.filterwarnings("ignore")
        np.random.seed(40)

        # unzip and  loadinf
        zipfolder = sys.argv[2] if len(sys.argv) > 2 else './titanic.zip'
        # zipfolder = './titanic.zip'
        filename = 'train.csv'
        raw_train = unzip_and_load(zipfolder, filename)

        # cleaning
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
        fill_cols = ['Age', 'Fare']
        train = clean_data(raw_train, drop_cols, fill_cols)

        # transform object to categorical to float
        obj_cols = train.select_dtypes(include='object').columns.to_list()
        print('object columns: ', obj_cols)

        for i in train.select_dtypes(include='object').columns.to_list():
            train[i] = train[i].astype('category')

        cater_cols = train.select_dtypes(include='category').columns.to_list()
        print('categorical columns: ', cater_cols)

        train['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

        # split into input (X) and label (Y) variables
        data_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        label_col = ['Survived']
        X_train = train[data_cols].astype(float)
        Y_train = train[label_col]

        # build model
        model = build_model(6, [128, 128, 1],
                            ['relu', 'relu', 'sigmoid'],
                            0, 1)
        model.summary()

        # model optimizer
        alpha = 0.005
        beta1 = 0.95
        beta2 = 0.8
        optimize_model(model, alpha, beta1, beta2)

        # training
        batch_size = 128

        mode = sys.argv[1] if len(sys.argv) > 1 else 'standard'
        if mode == 'debug':
            epochs = 1
        else:
            epochs = 300

    validation_split = 0.1

    train_model(model, X_train, Y_train, batch_size, epochs,
                early_stopping=True,
                patience=10, learning_rate_decay=False, alpha=alpha,
                save_best=True, filepath='network.h5', shuffle=True,
                validation_split=validation_split)

    # validation metrics
    print('\nValidation metrics')
    split_at = int(len(X_train) * (1 - validation_split))
    X_val = X_train[split_at:]
    Y_val = Y_train[split_at:]

    print('[val_loss, val_accuracy]', model.evaluate(X_val, Y_val))
