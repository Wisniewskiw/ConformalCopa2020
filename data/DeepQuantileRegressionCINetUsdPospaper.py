
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm  import  tqdm
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
from keras.layers import CuDNNLSTM
from keras import backend as K
from MyWrapper import *




class DeepQuantileRegressionWithUncertainty(object):
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

        if parameters_dict is None:
            parameters_dict = MyWrapper({})
        else:
            parameters_dict = MyWrapper(parameters_dict)

        self.df_path = parameters_dict.df_path


        self.TimeColumnName = parameters_dict.TimeColumnName
        self.rolling_window_size = parameters_dict.rolling_window_size
        self.columns_train = parameters_dict.columns_train
        self.lower_quantile_cutoff = parameters_dict.lower_quantile_cutoff
        self.higher_quantile_cutoff = parameters_dict.higher_quantile_cutoff

        self.quantile_or_mean = parameters_dict.quantile_or_mean
        self.epochs = parameters_dict.epochs
        self.train_test_split = parameters_dict.train_test_split
        self.batch_size = parameters_dict.batch_size


        self.custom_rate=parameters_dict.custom_rate
        if self.custom_rate:
            self.lr=parameters_dict.lr
            self.anealing_rate = parameters_dict.anealing_rate

        self.cyclical_rate = parameters_dict.cyclical_rate
        if self.cyclical_rate:
            self.cyclical_rate = parameters_dict.anealing_rate



        self.load_df()


    def load_df(self):
        '''

        :return:
        '''
        #self.df = pd.read_csv(self.df_path, parse_dates=[self.TimeColumnName], infer_datetime_format=True)
        self.df = pd.read_csv(self.df_path)


    @abc.abstractmethod
    def preprocess(self):
        '''

        :return:
        '''

    @abc.abstractmethod
    def postprocess(self,f=None):
        '''

        :param f:
        :return:
        '''


    def generate_index(self,cols):
        '''

        :return:
        '''

        data_matrix = self.df[cols]
        num_elements = data_matrix.shape[0]

        for start, stop in zip(range(0, num_elements - self.rolling_window_size, 1), range(self.rolling_window_size, num_elements, 1)):
            yield data_matrix[stop - self.rolling_window_size:stop].values.reshape((-1, len(cols)))


    def q_loss(self,q, y, f):
        e = (y - f)
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)






    def TrainLSTM(self):

         if self.cyclical_rate:
             self.history = self.model.fit(self.X_train, [self.y_train, self.y_train, self.y_train], callbacks=[clr], epochs=self.epochs,
                                      batch_size=self.batch_size, verbose=0, shuffle=False)



         else:
            self.history = self.model.fit( self.X_train, [self.y_train, self.y_train, self.y_train],  epochs=self.epochs,
                                          batch_size=self.batch_size, verbose=0, shuffle=False)


    def prediction_uncertainty(self,iterations=100):

        pred_ql, pred_50, pred_qh = [], [], []
        NN = K.function([self.model.layers[0].input, K.learning_phase()],
                        [self.model.layers[-3].output, self.model.layers[-2].output, self.model.layers[-1].output])

        for i in   (range(0, iterations)):
            predd = NN([self.X_test, 0.5])
            pred_ql.append(predd[0])
            pred_50.append(predd[1])
            pred_qh.append(predd[2])

        self.pred_ql = np.asarray(pred_ql)[:, :, 0]
        self.pred_50 = np.asarray(pred_50)[:, :, 0]
        self.pred_qh = np.asarray(pred_qh)[:, :, 0]

    def crossover_check(self):
        plt.figure(figsize=(35, 10))
        plt.plot(self.pred_qh, color='yellow',label='upper bound')
        plt.plot(self.pred_50, color='blue',label='median')
        plt.plot(self.pred_ql, color='green',label='lower bound')

        ### CROSSOVER CHECK ###
        plt.scatter(np.where(np.logical_or(self.pred_50 > self.pred_qh, self.pred_50 < self.pred_ql))[0],
                    self.pred_50[np.logical_or(self.pred_50 > self.pred_qh, self.pred_50 < self.pred_ql)],
                    c='red', s=50)
        plt.plot(self.y_test, color='black',label='actual y test')
        plt.legend()
        plt.show()

    def uncertainty_cylider(self):

        if self.quantile_or_mean:

            self.pred_ql = np.percentile(self.pred_ql, self.lower_quantile_cutoff, axis=0)
            self.pred_50 = self.pred_50.mean(axis=0)
            self.pred_qh = np.percentile(self.pred_qh, self.higher_quantile_cutoff, axis=0)
        else:
            self.pred_ql =  self.pred_ql.mean(axis=0)
            self.pred_50 = self.pred_50.mean(axis=0)
            self.pred_qh =  self.pred_qh.mean(axis=0)



    def evaluation_prediction(self   ):

        return self.loss_eval(self.y_test, self.pred_50)

    def quantile_range(self):
        plt.figure(figsize=(35, 10))
        plt.plot(self.y_test, color='red', alpha=0.4)
        plt.scatter( range(len(self.pred_ql)),self.pred_qh - self.pred_ql)
        plt.show()

    def proportion_contained_in_CI(self):
        bounds_df = pd.DataFrame()

        # Using 99% confidence bounds
        bounds_df['lower_bound'] = self.pred_ql
        bounds_df['prediction'] = self.pred_50
        bounds_df['real_value'] = self.y_test
        bounds_df['upper_bound'] = self.pred_qh

        bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                                  (bounds_df['real_value'] <= bounds_df['upper_bound']))

        print("Proportion of points contained within "+str(self.higher_quantile_cutoff-self.lower_quantile_cutoff)
              +"% confidence interval:",bounds_df['contained'].mean())


    def LSTM_model(self):

        losses = [lambda y, f: self.q_loss(self.lower_quantile_cutoff, y, f),
                  lambda y, f: self.q_loss(0.5, y, f),
                  lambda y, f: self.q_loss(self.higher_quantile_cutoff, y, f)]

        dropout1 = Dropout(0.15, name='droput1')
        dropout2 = Dropout(0.15, name='droput2')


        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))

        lstm = Bidirectional(CuDNNLSTM(64, return_sequences=True, name='lstm1'))(inputs)
        lstm = dropout1(lstm)
        lstm = Bidirectional(CuDNNLSTM(64, return_sequences=False, name='lstm1'))(lstm)
        lstm = dropout2(lstm)
        dense = Dense(50)(lstm)
        out10 = Dense(1)(dense)
        out50 = Dense(1)(dense)
        out90 = Dense(1)(dense)

        self.model = Model(inputs, [out10, out50, out90])

        if self.cyclical_rate:
            self.clr = CyclicLR(base_lr=0.001, max_lr=0.02,
                       step_size=10.)
            self.model.compile(loss=losses, optimizer='adam', loss_weights=[0.3, 0.3, 0.3])
        elif self.custom_rate:
            opt = keras.optimizers.Adagrad(learning_rate=0.01)
            self.model.compile(loss=losses, optimizer=opt, loss_weights=[0.3, 0.3, 0.3])

        else:
            self.model.compile(loss=losses, optimizer='adam', loss_weights=[0.3, 0.3, 0.3])

    def RUN(self):
        #print('preprocess')
        self.preprocess()
        #print('model')
        self.LSTM_model()
        self.TrainLSTM()
        #print('prediction uncertainty')
        self.prediction_uncertainty()
        #print('uncertainty cylinder')
        self.uncertainty_cylider()
        #print('postprocess')
        self.postprocess()


        #self.crossover_check()
        #self.quantile_range()

        #print(self.evaluation_prediction())
        #self.proportion_contained_in_CI()





class NetPosUsdDQRWU(DeepQuantileRegressionWithUncertainty):

    def __init__(self,parameters_dict):

        super(NetPosUsdDQRWU, self).__init__(parameters_dict)
        self.loss_eval=lambda x,y:mean_squared_log_error(x,y)

    def mad(self,arr):
        """ Median Absolute Deviation: a "Robust" version of standard deviation.
            Indices variabililty of the sample.
            https://en.wikipedia.org/wiki/Median_absolute_deviation
        """
        arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
        med = np.median(arr)
        return np.median(np.abs(arr - med))

    def mlog_trans(self,x):

        x_trans = (1 / mad(x)) * (x - np.median(x))
        y = np.sign(x_trans) * (np.log(abs(x_trans) + 1 / (1 / 3)) + np.log(1 / 3))
        return y

    def mlog_inverse(self,x_trans, median_x, mad_x):
        y = np.sign(x_trans) * (np.exp(abs(x_trans) - np.log(1 / 3)) - 1 / (1 / 3))
        x_inv = mad_x * y + median_x
        return x_inv

    def no_bardata_nans(self):
        return self.df[np.isnan(self.df.CloseBid) == False]

    def preprocess(self ):


        self.mad_=self.mad(self.df.NetPosUsd)
        self.median = np.median(self.df.NetPosUsd)
        self.df.NetPosUsd=self.mlog_trans(self.df.NetPosUsd)
        self.df['ma'+str(self.rolling_window_size/3)] = self.df['NetPosUsd'].rolling(window=int(self.rolling_window_size/3)).mean()
        self.df['ma'+str(self.rolling_window_size)] = self.df['NetPosUsd'].rolling(window=self.rolling_window_size).mean()
        other = []

        cnt, mean = [], []
        for sequence in self.generate_index(self.columns_train):
            cnt.append(sequence)

        cnt = np.array(cnt)

        # out = np.concatenate([cnt, other], axis=2)
        out = cnt

        label = self.df.NetPosUsd[self.rolling_window_size:].values
        self.X_train, self.X_test = out[:self.train_test_split], out[self.train_test_split:]
        self.y_train, self.y_test = label[:self.train_test_split], label[self.train_test_split:]







    def postprocess(self):
        self.pred_qh = self.mlog_inverse(self.pred_qh ,self.median,self.mad_ )
        self.pred_50 = self.mlog_inverse(self.pred_50  ,self.median,self.mad_ )
        self.pred_ql = self.mlog_inverse(self.pred_ql  ,self.median,self.mad_ )
        self.y_test = self.mlog_inverse(self.y_test,self.median,self.mad_ )




















def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))



def mlog_trans(x):

  x_trans= (1/mad(x))*(x-np.median(x))
  y = np.sign(x_trans)*(np.log(abs(x_trans)+1/(1/3))+np.log(1/3))
  return y


def mlog_inverse(x_trans,median_x, mad_x):
  y =np.sign(x_trans)*(np.exp(abs(x_trans)-np.log(1/3))-1/(1/3))
  x_inv = mad_x*y+median_x
  return x_inv

def no_bardata_nans(df):
    return df[np.isnan(df.CloseBid)==False]

def cylinder_log_hist(df):
    d = df['Upper_60_Cylinder'] - df['Lower_60_Cylinder']
    plt.hist(np.log(d[d > 0]), bins=25);
    plt.show()
    d = df['Upper_360_Cylinder'] - df['Lower_360_Cylinder']
    plt.hist(np.log(d[d > 0]), bins=25);
    plt.show()
    d = df['Upper_1440_Cylinder'] - df['Lower_1440_Cylinder']
    plt.hist(np.log(d[d > 0]), bins=25);
    plt.show()
    d = df['Upper_8640_Cylinder'] - df['Lower_8640_Cylinder']
    plt.hist(np.log(d[d > 0]), bins=25);
    plt.show()


def prop(Lower_Cylinder,Upper_Cylinder, NetPosUsd,CI):
    bounds_df = pd.DataFrame()

    # Using CI% confidence bounds
    bounds_df['lower_bound'] = df[Lower_Cylinder]

    bounds_df['real_value'] = df[NetPosUsd]
    bounds_df['upper_bound'] = df[Upper_Cylinder]

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                              (bounds_df['real_value'] <= bounds_df['upper_bound']))

    print("Proportion of points contained within "+CI+ "% confidence interval:",bounds_df['contained'].mean())

def trainqlstmalpha(a):
    import json
    with open('DQhourly.json', 'r') as fp:
        parameters_dict  = json.load(fp)
    alpha = a
    parameters_dict_real_taxi['lower_quantile_cutoff'] = alpha / 2
    parameters_dict_real_taxi['higher_quantile_cutoff'] = 1 - alpha / 2
    for i in tqdm(range(29)):

        parameters_dict_real_taxi['rolling_window_size'] = 48

        parameters_dict_real_taxi['train_test_split'] = 14323 - parameters_dict_real_taxi['rolling_window_size']
        parameters_dict_real_taxi['df_path'] = 'EURUSD_NETPOSUSD_hourly' + str(i) + '.csv'
        DQR = NetPosUsdDQRWU(parameters_dict_real_taxi)
        DQR.RUN()

        d = {'QdfTime': DQR.df[-120:].QdfTime,
             'NetPosUsd': DQR.y_test,
             'prediction': DQR.pred_50,
             'QR_lower': DQR.pred_ql,

             'QR_upper': DQR.pred_qh

             }
        out = pd.DataFrame(data=d)

        if i == 0:
            out.to_csv('QuantileLstmBoundsBiLSTMCuda_alpha' + str(int(alpha * 100)) + '.csv', encoding='utf-8',
                       index=False)
        else:
            out.to_csv('QuantileLstmBoundsBiLSTMCuda_alpha' + str(int(alpha * 100)) + '.csv', mode='a', header=False,
                       index=False)
        del DQR


def trainqlstm():
    for a in tqdm(np.linspace(5, 95, 19)):
        trainqlstmalpha(a)
