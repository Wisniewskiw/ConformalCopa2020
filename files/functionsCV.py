import numpy as np
import pandas as pd
from scipy import stats

from tqdm import tqdm
import os.path

from sklearn.ensemble import GradientBoostingRegressor
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from lightgbm import LGBMRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.icp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, RegressorNormalizer

import keras

import tensorflow as tf
import random as rn
import itertools

np.random.seed(42)
rn.seed(12345)
from keras import backend as K

################################################################################
import logging
import copy

import os
import keras

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout



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
        self.rolling_window_size=parameters_dict.get('window')
        self.neurons_layer1 = parameters_dict.get('neurons_layer1')
        self.dropout_p = parameters_dict.get('dropout_p')
        self.epochs = parameters_dict.get('epochs')
        self.batch_size = parameters_dict.get('batch_size')





    def generate_index(self, data_matrix):
        '''

        :return:
        '''


        num_elements = data_matrix.shape[0]

        for start, stop in zip(range(0, num_elements - self.rolling_window_size, 1),
                               range(self.rolling_window_size, num_elements, 1)):
            yield data_matrix[stop - self.rolling_window_size:stop].values.reshape((-1, data_matrix.shape[1]))


    def q_loss(self,q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    def LSTM_forcaster_model(self):

        dropout1 = Dropout(self.dropout_p,name='droput1')
        dropout2 = Dropout(self.dropout_p,name='droput2')

        inputs = Input(shape=(self.X_train_ext.shape[1], self.X_train_ext.shape[2]))

        lstm = Bidirectional(CuDNNLSTM(self.neurons_layer1, return_sequences=True,name='lstm1'))(inputs)

        lstm= dropout1(lstm)
        lstm = Bidirectional(CuDNNLSTM(16, return_sequences=False,name='lstm2'))(lstm)

        lstm = dropout2(lstm)

        dense = Dense(50,name='bigdense')(lstm)
        out = Dense(1,name='output')(dense)
        self.model = Model(inputs, out)
        self.model.compile(loss=lambda y, f: self.q_loss(0.5, y, f), optimizer='adam')

    def preprocess(self):

        cnt, mean = [], []
        for sequence in self.generate_index(self.columns_train):
            cnt.append(sequence)

        cnt = np.array(cnt)

        # out = np.concatenate([cnt, other], axis=2)
        out = cnt

        # ext = []
        # for sequence in self.generate_index(['OpenAsk']):
        #    ext.append(sequence)

        label = self.df.NetPosUsd[self.rolling_window_size:].values
        self.X_train, self.X_test = out[:self.train_test_split], out[self.train_test_split:]
        self.y_train, self.y_test = label[:self.train_test_split], label[self.train_test_split:]

    def fit(self,X,y):

        cnt=[]
        for sequence in self.generate_index(X):
            cnt.append(sequence)

        cnt = np.array(cnt)

        self.history = self.model.fit(cnt, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=False)




    def predict(self,X,iterations=100):

        cnt = []
        for sequence in self.generate_index(X):
            cnt.append(sequence)

        cnt = np.array(cnt)

        pred_50  =   []
        NN = K.function([self.model.layers[0].input, K.learning_phase()],
                          [self.model.layers[-1].output])

        for i in  (range(0, iterations)):
            predd = NN([cnt, 0.5])
            pred_50.append(predd[0])

        self.pred_50 = np.asarray(pred_50)[:, :, 0]
        self.prediction = self.pred_50.mean(axis=0)

        return  self.prediction





class NeuralNetworkAlgorithm(object):

    def __init__(self, params):

        self.learner_params = {
            "dense_layers": params.get("dense_layers"),
            "dense_1_size": params.get("dense_1_size"),
            "dense_2_size": params.get("dense_2_size"),
            "dense_3_size": params.get("dense_3_size"),
            "dropout": params.get("dropout"),
            "rounds": params.get("rounds")
        }
        self.model = None  # we need input data shape to construct model

    def create_model(self, input_dim):
        self.model = Sequential()
        for i in range(self.learner_params.get("dense_layers")):
            self.model.add(
                Dense(
                    self.learner_params.get("dense_{}_size".format(i + 1)),
                    activation="relu",
                    input_dim=input_dim,
                )
            )
            if self.learner_params.get("dropout"):
                self.model.add(Dropout(rate=self.learner_params.get("dropout")))

        self.model.add(Dense(1))

        self.model.compile(
            optimizer='adam', loss="mae", metrics=["mae"])

    def update(self, update_params):
        pass

    def fit(self, X, y):

        if self.model is None:
            self.create_model(input_dim=X.shape[1])

        self.model.fit(X, y, batch_size=256, epochs=self.learner_params.get("rounds"), verbose=False)

    def predict(self, X):

        return np.ravel(self.model.predict(X))


# define loss fn
def qd_objective(y_true, y_l, y_u, alpha__):
    lambda_ = 0.01  # lambda in loss fn

    soften_ = 160.
    n_ = len(y_true)  # batch size
    '''Loss_QD-soft, from algorithm 1'''

    K_HU = np.maximum(0., np.sign(y_u - y_true))
    K_HL = np.maximum(0., np.sign(y_true - y_l))
    K_H = K_HU * K_HL

    K_SU = expit(soften_ * (y_u - y_true))
    K_SL = expit(soften_ * (y_true - y_l))

    K_S = K_SU * K_SL

    MPIW_c = np.sum(((y_u - y_l) * K_H)) / np.sum(K_H)
    PICP_H = np.mean(K_H)
    PICP_S = np.mean(K_S)

    Loss_S = MPIW_c + lambda_ * n_ / (alpha__ * (1 - alpha__)) * np.maximum(0., (1 - alpha__) - PICP_H)
    Loss_SN = MPIW_c / np.max(((y_u - y_l) * K_H)) + lambda_ * n_ / (alpha__ * (1 - alpha__)) * np.maximum(0., (
            1 - alpha__) - PICP_H)

    return Loss_SN


def grid_cv(df, parameters):
    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for params in tqdm(permutations_dicts):
        cv(df, params)


def cv(df, parameters):
    end = len(df) - 120
    out = np.zeros(3)
    out2 = np.zeros(3)
    p = parameters.copy()
    p.pop('algorithm')
    p.pop('randomized_calibration')
    p.pop('alpha_')
    if parameters.get('algorithm') == 'RandomForest':
        algorithm = RandomForestRegressor(**p)
        d = {'n_estimators': parameters.get('n_estimators'),
             "criterion": parameters.get("criterion"),
             "max_features": parameters.get("max_features"),
             "min_samples_split": parameters.get("min_samples_split"),
             "min_samples_leaf": parameters.get("min_samples_leaf")
             }
    if parameters.get('algorithm') == 'K-NearestNeighbours':
        algorithm = KNeighborsRegressor(**p)
        d = {
            'n_neighbours': parameters.get('n_neighbours'),
            'weights': parameters.get('weights'),
            'metric': parameters.get('metric')
        }
    if parameters.get('algorithm') == 'LightGBM':
        algorithm = LGBMRegressor(**p)
        d = {"metric": parameters.get("metric"),
             "num_leaves": parameters.get('num_leaves'),
             "learning_rate": parameters.get('learning_rate'),
             "feature_fraction": parameters.get('feature_fraction'),
             "bagging_fraction": parameters.get('bagging_fraction'),
             "bagging_freq": parameters.get('bagging_freq'),
             }

    if parameters.get('algorithm') == 'LassoRegression':
        algorithm = Lasso(**p)
        d = {'alpha_': parameters.get('alpha_')}

    if parameters.get('algorithm') == 'NeuralNetwork':
        algorithm = NeuralNetworkAlgorithm(p)

    if parameters.get('algorithm') == 'LSTM':
        algorithm = BiLSTM(**p)
        d = {}
    d = p
    d['alpha_'] = parameters.get('alpha_')

    m, s = df['NetPosUsd'].mean(), df['NetPosUsd'].std()
    df=df.drop(['QdfTime' ], axis=1)
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    df = (df - mean) / std

    for i, ratio in enumerate(([.5, 0.66, .84])):
        if parameters.get('randomized_calibration') == True:

            train_ = df.drop([  'NetPosUsd'], axis=1).iloc[:int(end * ratio), :].values
            choose = np.random.choice(len(train_), int(end / 6), replace=False)
            calibrate = train_[choose, :]
            mask = np.ones(len(train_), dtype=bool)
            mask[choose] = False

            train = train_[mask, :]
            test = (df.drop([  'NetPosUsd'], axis=1)).iloc[int(end * ratio):int(end * ratio) + int(end / 6),
                   :].values

            ytrain_ = df['NetPosUsd'][:int(end * ratio)].values

            ycalibrate = ytrain_[choose]
            ytrain = ytrain_[mask]

            ytest = df['NetPosUsd'].iloc[int(end * ratio):int(end * ratio) + int(end / 6)]

        else:
            train = df.drop([  'NetPosUsd'], axis=1).iloc[:int(end * ratio) - int(end / 6), :].values

            calibrate = df.drop([  'NetPosUsd'], axis=1).iloc[int(end * ratio) - int(end / 6):int(end * ratio),
                        :].values

            test = df.drop([  'NetPosUsd'], axis=1).iloc[int(end * ratio):int(end * ratio) + int(end / 6),
                   :].values

            ytrain = df['NetPosUsd'][:int(end * ratio) - int(end / 6)].values

            ycalibrate = df['NetPosUsd'][int(end * ratio) - int(end / 6):int(end * ratio)].values

            ytest = df['NetPosUsd'][int(end * ratio):int(end * ratio) + int(end / 6)].values
            # print(len(train),len(ytrain),len(calibrate),len(ycalibrate),len(test),len(ytest))

            # Train and calibrate
        # -----------------------------------------------------------------------------

        underlying_model = RegressorAdapter(algorithm)
        normalizing_model = RegressorAdapter(KNeighborsRegressor(n_neighbors=50))
        normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())
        nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)

        icp = IcpRegressor(nc)
        icp.fit(train, ytrain)
        icp.calibrate(calibrate, ycalibrate)

        # -----------------------------------------------------------------------------
        # Predict
        # -----------------------------------------------------------------------------
        prediction = icp.predict(test, significance=parameters.get('alpha_'))
        header = ['NCP_lower', 'NCP_upper', 'NetPosUsd', 'prediction']
        size = prediction[:, 1] / 2 + prediction[:, 0] / 2

        prediction = prediction * s + m
        ytest = ytest * s + m
        size = size * s + m

        table = np.vstack([prediction.T, ytest, size.T]).T

        dfncp = pd.DataFrame(table, columns=header)

        underlying_model = RegressorAdapter(algorithm)

        nc = RegressorNc(underlying_model, AbsErrorErrFunc())
        icp = IcpRegressor(nc)
        icp.fit(train, ytrain)
        icp.calibrate(calibrate, ycalibrate)

        prediction = icp.predict(test, significance=parameters.get('alpha_'))
        header = ['cp_lower', 'cp_upper']

        prediction = prediction * s + m

        table = np.vstack([prediction.T]).T

        dfcp = pd.DataFrame(table, columns=header)
        dfncp['CP_lower'] = dfcp['cp_lower']
        dfncp['CP_upper'] = dfcp['cp_upper']

        out[i] = qd_objective(dfncp.NetPosUsd, dfncp['CP_lower'], dfncp['CP_upper'], parameters.get('alpha_'))

        out2[i] = qd_objective(dfncp.NetPosUsd, dfncp['NCP_lower'], dfncp['NCP_upper'], parameters.get('alpha_'))

    d['CP_loss'] = np.mean(out)
    d['NCP_loss'] = np.mean(out2)

    if os.path.exists(parameters.get('algorithm') + '_cv.csv') == True:

        pd.DataFrame(data=d, index=[0]).to_csv(parameters.get('algorithm') + '_cv.csv', mode='a', header=False,
                                               index=False)

    else:
        pd.DataFrame(data=d, index=[0]).to_csv(parameters.get('algorithm') + '_cv.csv', encoding='utf-8', index=False)


def train_and_test_cp_algo(parameters):
    p = parameters.copy()
    p.pop('algorithm')
    p.pop('randomized_calibration')
    p.pop('alpha_')
    p.pop('calibration_size')
    p.pop('WhichCP')

    for i in tqdm(range(29)):
        if parameters.get('algorithm') == 'RandomForest':
            algorithm = RandomForestRegressor(**p)
        if parameters.get('algorithm') == 'K-NearestNeighbours':
            algorithm = KNeighborsRegressor(**p)
        if parameters.get('algorithm') == 'LightGBM':
            algorithm = LGBMRegressor(**p)
        if parameters.get('algorithm') == 'LassoRegression':
            algorithm = Lasso(**p)
        if parameters.get('algorithm') == 'NeuralNetwork':
            algorithm = NeuralNetworkAlgorithm(p)
        if parameters.get('algorithm') == 'LSTM':
            algorithm = BiLSTM(**p)
        if parameters.get('algorithm') == 'GradientBoosting':
            algorithm =GradientBoostingRegressor(**p)


        path = 'data\EURUSD_NETPOSUSD_hourly_for_regresion' + str(i) + '.csv'
        df = pd.read_csv(path).drop(['Unnamed: 0','QdfTime'], axis=1).fillna(0)
        m, s = df['NetPosUsd'].mean(), df['NetPosUsd'].std()

        mean = df.mean(axis=0)
        std = df.std(axis=0)
        df = (df - mean) / std

        if parameters.get('randomized_calibration') == True:

            train_test_split = len(df) - 120
            train_ = df.drop([ 'NetPosUsd'], axis=1).iloc[:train_test_split, :].values
            choose = np.random.choice(len(train_), parameters.get("calibration_size"), replace=False)
            calibrate = train_[choose, :]
            mask = np.ones(len(train_), dtype=bool)
            mask[choose] = False
            train = train_[mask, :]

            test = (df.drop([  'NetPosUsd'], axis=1)).iloc[train_test_split:,
                   :].values

            ytrain_ = df['NetPosUsd'][:train_test_split].values

            ycalibrate = ytrain_[choose]
            ytrain = ytrain_[mask]

            ytest = df['NetPosUsd'].iloc[train_test_split:]


        else:
            train_test_split = len(df) - 120 - parameters.get("calibration_size")
            train = df.drop([  'NetPosUsd'], axis=1).iloc[:train_test_split, :].values

            calibrate = df.drop([ 'NetPosUsd'], axis=1).iloc[train_test_split:train_test_split + parameters.get("calibration_size"), :].values

            test = (df.drop([  'NetPosUsd'], axis=1)).iloc[-120:,:].values

            ytrain = df['NetPosUsd'][:train_test_split].values

            ycalibrate = df['NetPosUsd'][train_test_split:train_test_split + parameters.get("calibration_size")]

            ytest = df['NetPosUsd'].iloc[-120:]

        if parameters.get("WhichCP") == 'NCP':
            underlying_model = RegressorAdapter(algorithm)
            normalizing_model = RegressorAdapter(KNeighborsRegressor(n_neighbors=50))
            normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())
            nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)
            icp = IcpRegressor(nc)
            icp.fit(train, ytrain)
            icp.calibrate(calibrate, ycalibrate)

            # -----------------------------------------------------------------------------
            # Predict
            # -----------------------------------------------------------------------------
            prediction = icp.predict(test, significance=parameters.get('alpha_'))
            header = ['NCP_lower', 'NCP_upper', 'NetPosUsd', 'prediction']
            size = prediction[:, 1] / 2 + prediction[:, 0] / 2

            prediction=prediction*s+m
            ytest=ytest*s+m
            size=size*s+m

            table = np.vstack([prediction.T, ytest, size.T]).T

            dfncp = pd.DataFrame(table, columns=header)

        else:
            underlying_model = RegressorAdapter(algorithm)
            nc = RegressorNc(underlying_model, AbsErrorErrFunc())
            icp = IcpRegressor(nc)
            icp.fit(train, ytrain)
            icp.calibrate(calibrate, ycalibrate)

            # -----------------------------------------------------------------------------
            # Predict
            # -----------------------------------------------------------------------------
            prediction = icp.predict(test, significance=parameters.get('alpha_'))
            header = ['CP_lower', 'CP_upper', 'NetPosUsd', 'prediction']
            size = prediction[:, 1] / 2 + prediction[:, 0] / 2

            prediction = prediction * s + m
            ytest = ytest * s + m
            size = size * s + m

            table = np.vstack([prediction.T, ytest, size.T]).T

            dfncp = pd.DataFrame(table, columns=header)

        if i == 0:
            dfncp.to_csv(
                parameters.get("WhichCP") + '_' + parameters.get('algorithm') + '_' + str(
                    np.round(parameters.get('alpha_') * 100).astype(int)) + '_' + 'calibrationwindow' + str(
                    parameters.get('calibration_size')) + '.csv',
                encoding='utf-8', index=False)
        else:
            dfncp.to_csv(
                parameters.get("WhichCP") + '_' + parameters.get('algorithm') + '_' + str(
                    np.round(parameters.get('alpha_') * 100).astype(int)) + '_' + 'calibrationwindow' + str(
                    parameters.get('calibration_size')) + '.csv', mode='a',
                header=False, index=False)

        del algorithm


def all_pvalues_algo(algorithm, WhichCP, calibration_size, calibration=True, hypertuning=True):
    if hypertuning:
        df = pd.read_csv(algorithm + '_cv.csv')

        if WhichCP == 'CP':
            CP_dicts = df.loc[df.groupby('alpha_').CP_loss.idxmin()]
            CP_dicts = CP_dicts.drop(['NCP_loss', 'CP_loss'], axis=1).to_dict(orient='records')
        else:
            CP_dicts = df.loc[df.groupby('alpha_').NCP_loss.idxmin()]
            CP_dicts = CP_dicts.drop(['NCP_loss', 'CP_loss'], axis=1).to_dict(orient='records')

    else:
        CP_dicts = dict()
        if algorithm == 'K-NearestNeighbours':
            CP_dicts = {'n_neighbors': 200, 'n_jobs': -1, 'weights': 'distance'}

        elif algorithm == 'LassoRegression':
            CP_dicts = {'alpha': 1.0}
        elif algorithm == 'RandomForest':
            CP_dicts = {'min_samples_leaf': 5}
        elif algorithm == 'LightGBM':
            CP_dicts = {'min_samples_leaf': 5}
        elif algorithm == 'NeuralNetwork':
            CP_dicts = {"dense_layers": 3,
                        "dense_1_size": 32,
                        "dense_2_size": 16,
                        "dense_3_size": 8,
                        "dropout": 0.5, 'rounds': 50}

        elif  algorithm == 'GradientBoosting':
            CP_dicts = {"min_samples_leaf": 5}
        else:
            CP_dicts = {}
        out=[]
        print(CP_dicts)
        for a in np.linspace(0.05,.95,19):
            d=CP_dicts.copy()
            d['alpha_']=a
            out.append(d)
        CP_dicts=out
    

 
    for dictio in tqdm(CP_dicts):
        dictio['algorithm'] = algorithm
        dictio['WhichCP'] = WhichCP
        dictio['randomized_calibration'] = calibration
        dictio['calibration_size'] = calibration_size
        train_and_test_cp_algo(dictio)


def all_calibration_windows_algo(algorithm, WhichCP, alpha=0.05, calibration=True, hypertuning=True):
    if hypertuning == True:
        df = pd.read_csv(algorithm + '_cv.csv')
        df = df[df.alpha_ == alpha]

        if WhichCP == 'CP':
            CP_dicts = df.loc[df.groupby('alpha_').CP_loss.idxmin()]
            CP_dicts = CP_dicts.drop(['NCP_loss', 'CP_loss'], axis=1).to_dict(orient='records')[0]
        else:
            CP_dicts = df.loc[df.groupby('alpha_').NCP_loss.idxmin()]
            CP_dicts = CP_dicts.drop(['NCP_loss', 'CP_loss'], axis=1).to_dict(orient='records')[0]
    else:
        CP_dicts = dict()
        if algorithm == 'K-NearestNeighbours':
            CP_dicts = {'n_neighbors': 120, 'n_jobs': -1, 'weights': 'distance'}

        elif algorithm == 'LassoRegression':
            CP_dicts = {'alpha': 1.0}
        elif algorithm == 'RandomForest':
            CP_dicts = {'min_samples_leaf': 5}
        elif algorithm == 'LightGBM':
            CP_dicts = {'min_samples_leaf': 5}
        elif algorithm == 'NeuralNetwork':
            CP_dicts = {"dense_layers": 3,
                        "dense_1_size": 32,
                        "dense_2_size": 16,
                        "dense_3_size": 8,
                        "dropout": 0.5, 'rounds': 50}
        elif  algorithm == 'GradientBoosting':
            CP_dicts = {"min_samples_leaf": 5}
        else:
            CP_dicts = {}
        out=[]
        for a in np.linspace(0.05,.95,19):
            d=CP_dicts.copy()
            d['alpha_']=a
            out.append(d)
        CP_dicts=out
    
    

    for calib_size in tqdm(
            [50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000]):
        CP_dicts['algorithm'] = algorithm
        CP_dicts['WhichCP'] = WhichCP
        CP_dicts['alpha_'] =alpha
        CP_dicts['randomized_calibration'] = calibration
        CP_dicts['calibration_size'] = calib_size
        train_and_test_cp_algo(CP_dicts)

