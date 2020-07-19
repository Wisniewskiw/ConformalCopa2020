from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer

import random as rn
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import re
import datetime
import abc
from abc import ABCMeta
from keras.models import *
from keras.layers import *
from keras.layers.core import Lambda
from keras import backend as K

from keras import backend as K

from keras.layers import CuDNNLSTM
from keras.layers import Dropout
import tensorflow as tf


class BiLSTM(object):
    '''
    Deep keras quantile regression with uncertainty.

    1) lstm  model is defined
    2) the model is trained on rolling window basis
    3)

    '''
    __metaclass__ = ABCMeta

    def __init__(self, parameters_dict):
        '''

        :param parameters_dict: dict with many parameters

        '''
        self.rolling_window_size = parameters_dict.get('window')

    def q_loss(self, q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    def fit(self, X, y):
        XX = X.reshape(X.shape[0], X.shape[1], 1)
        dropout1 = Dropout(0.3, name='droput1')
        dropout2 = Dropout(0.3, name='droput2')

        inputs = Input(shape=(XX.shape[1], XX.shape[2]))

        lstm = Bidirectional(CuDNNLSTM(64, return_sequences=True, name='lstm1'))(inputs)

        lstm = dropout1(lstm)
        lstm = Bidirectional(CuDNNLSTM(16, return_sequences=False, name='lstm2'))(lstm)

        lstm = dropout2(lstm)

        dense = Dense(50, name='bigdense')(lstm)
        out = Dense(1, name='output')(dense)
        self.model = Model(inputs, out)
        self.model.compile(loss=lambda y, f: self.q_loss(0.5, y, f), optimizer='adam')

        self.history = self.model.fit(XX, y, epochs=50, batch_size=128, verbose=0, shuffle=False)

    def predict(self, X, iterations=100):
        XX = X.reshape(X.shape[0], X.shape[1], 1)
        pred_50 = []
        NN = K.function([self.model.layers[0].input, K.learning_phase()],
                        [self.model.layers[-1].output])

        for i in (range(0, iterations)):
            predd = NN([XX, 0.5])
            pred_50.append(predd[0])

        self.pred_50 = np.asarray(pred_50)[:, :, 0]
        self.prediction = self.pred_50.mean(axis=0)

        return self.prediction


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def mlog_trans(x):
    x_trans = (1 / mad(x)) * (x - np.median(x))
    y = np.sign(x_trans) * (np.log(abs(x_trans) + 1 / (1 / 3)) + np.log(1 / 3))
    return y


def mlog_inverse(x_trans, median_x, mad_x):
    y = np.sign(x_trans) * (np.exp(abs(x_trans) - np.log(1 / 3)) - 1 / (1 / 3))
    x_inv = mad_x * y + median_x
    return x_inv


def train_and_test_cp_algo(i):
    window = 96
    p = {'window': window}
    algorithm = BiLSTM(p)

    path = 'data\EURUSD_NETPOSUSD_hourly_for_regresion' + str(i) + '.csv'
    df = pd.read_csv(path).drop(['QdfTime', 'Unnamed: 0'], axis=1).fillna(0)
    y_raw_test = df.NetPosUsd[-120:]
    median_ = df.NetPosUsd.median()
    mad_ = mad(df.NetPosUsd.values)
    df.NetPosUsd = mlog_trans(df.NetPosUsd.values)

    # mean = df.NetPosUsd.mean()
    # std = df.NetPosUsd.std()
    # df.NetPosUsd = (df.NetPosUsd - mean) / std

    data = df.NetPosUsd.values

    def generate_index(window, data_matrix):
        '''

        :return:
        '''

        num_elements = data_matrix.shape[0]

        for start, stop in zip(range(0, num_elements - window, 1), range(window, num_elements, 1)):
            yield data_matrix[stop - window:stop].reshape((-1, 1))

    cnt = []

    for sequence in generate_index(window, data):
        cnt.append(sequence)
    cnt = np.array(cnt)

    X = cnt
    y = data[window:]

    X = X.reshape(X.shape[0], X.shape[1])

    train_test_split = X.shape[0] - 120 - 3480
    train = X[:train_test_split, :]

    calibrate = X[train_test_split:train_test_split + 3480, :]

    test = X[-120:]

    ytrain = y[:train_test_split]

    ycalibrate = y[train_test_split:train_test_split + 3480]

    ytest = y[-120:]

    underlying_model = RegressorAdapter(algorithm)
    normalizing_model = RegressorAdapter(KNeighborsRegressor(n_neighbors=50))
    normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())
    nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)
    icp = IcpRegressor(nc)
    icp.fit(train, ytrain)
    icp.calibrate(calibrate, ycalibrate)

    underlying_model2 = RegressorAdapter(algorithm)
    nc2 = RegressorNc(underlying_model2, AbsErrorErrFunc())
    icp2 = IcpRegressor(nc2)
    icp2.fit(train, ytrain)
    icp2.calibrate(calibrate, ycalibrate)

    for a in tqdm(np.linspace(5, 95, 19)):

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp.predict(test, significance=a / 100)
        header = ['NCP_lower', 'NCP_upper', 'NetPosUsd', 'prediction']
        lower, upper = prediction[:, 0], prediction[:, 1]

        lower = mlog_inverse(lower, median_, mad_)
        upper = mlog_inverse(upper, median_, mad_)
        ytest = mlog_inverse(ytest, median_, mad_)
        # lower=lower*std+mean
        # upper=upper*std+mean
        # ytest=ytest*std+mean
        size = upper / 2 + lower / 2
        table = np.vstack([lower, upper, y_raw_test, size.T]).T

        dfncp = pd.DataFrame(table, columns=header)

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp2.predict(test, significance=a / 100)
        header = ['CP_lower', 'CP_upper', 'NetPosUsd', 'prediction']
        lower, upper = prediction[:, 0], prediction[:, 1]

        lower = mlog_inverse(lower, median_, mad_)
        upper = mlog_inverse(upper, median_, mad_)
        ytest = mlog_inverse(ytest, median_, mad_)

        # lower=lower*std+mean
        # upper=upper*std+mean
        # ytest=ytest*std+mean
        size = upper / 2 + lower / 2
        table = np.vstack([lower, upper, y_raw_test, size.T]).T

        dfcp = pd.DataFrame(table, columns=header)

        if i == 0:
            dfcp.to_csv(
                'CP' + '_' + 'cudaLSTM' + '_' + str(
                    np.round(a).astype(int)) + '_' + 'calibrationwindow' + str(
                    3480) + '.csv',
                encoding='utf-8', index=False)
        else:
            dfcp.to_csv(
                'CP' + '_' + 'cudaLSTM' + '_' + str(
                    np.round(a).astype(int)) + '_' + 'calibrationwindow' + str(
                    3480) + '.csv', mode='a',
                header=False, index=False)

        if i == 0:
            dfncp.to_csv(
                'NCP' + '_' + 'cudaLSTM' + '_' + str(
                    np.round(a).astype(int)) + '_' + 'calibrationwindow' + str(
                    3480) + '.csv',
                encoding='utf-8', index=False)
        else:
            dfncp.to_csv(
                'NCP' + '_' + 'cudaLSTM' + '_' + str(
                    np.round(a).astype(int)) + '_' + 'calibrationwindow' + str(
                    3480) + '.csv', mode='a',
                header=False, index=False)

def trainlstm():
    for i in tqdm(range(29)):
        train_and_test_cp_algo(i)







