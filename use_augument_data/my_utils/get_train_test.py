from preprocessing import augumentSimple
from sklearn.model_selection import train_test_split
from use_augument_data.hyper_tunnning import *
import numpy as np


def get_row_data(path):
    X, Y = augumentSimple.preprocess(path,
                                     data_mark,
                                     fs,
                                     win_tlen,
                                     overlap_rate,
                                     random_seed,
                                     norm=3)
    return X, Y;


def get_train_test(path):
    X, Y = get_row_data(path)
    print(X.shape)
    X = X.reshape(X.shape[0], 32, 32, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    return x_train, x_test, y_train, y_test


def get_train_test_1dim(path):
    X, Y = get_row_data(path)
    print(X.shape)
    X = X.reshape(X.shape[0], 1024)
    X, Y = X[:, :, np.newaxis], Y[:, np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    return x_train, x_test, y_train, y_test
