import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc


def train_and_test_quantile_QCP(parameters):
    params = parameters.copy()
    params.pop('algorithm')
    quantiles_forest = [(params['alpha_'] / 2), (100 - params['alpha_'] / 2)]
    params.pop('alpha_')
    validation = params['validation']
    params.pop('validation')

    for i in tqdm(range(29)):

        path = 'data\EURUSD_NETPOSUSD_hourly_for_regresion' + str(i) + '.csv'
        df = pd.read_csv(path).drop(['Unnamed: 0', 'QdfTime'], axis=1).fillna(0)
        train_test_split = len(df) - 120
        m, s = df['NetPosUsd'].mean(), df['NetPosUsd'].std()

        mean = df.mean(axis=0)
        std = df.std(axis=0)
        df = (df - mean) / std

        train_test_split = len(df) - 120
        train = 1 * df.drop(['NetPosUsd'], axis=1).iloc[:train_test_split, :].values
        test = 1 * (df.drop(['NetPosUsd'], axis=1)).iloc[train_test_split:, :].values

        ytrain = df['NetPosUsd'][:train_test_split].values
        ytest = df['NetPosUsd'].iloc[train_test_split:]

        idx_train = np.arange(train_test_split - validation)
        idx_cal = np.arange(train_test_split - validation, train_test_split)

        if parameters.get('algorithm') == 'QuantileGradientBoosting':
            quantile_estimator = helper.QuantileGradientBoosting(model=None, quantiles=quantiles_forest, params=params)

        if parameters.get('algorithm') == 'QuantileLightGBM':
            quantile_estimator = helper.QuantileLightGBM(model=None, quantiles=quantiles_forest, params=params)
        if parameters.get('algorithm') == 'QuantileRegression':
            quantile_estimator = helper.QuantileRegression(model=None, quantiles=quantiles_forest, params=params)
        if parameters.get('algorithm') == 'QuantileRandomForest':
            quantile_estimator = helper.QuantileForestRegressorAdapterNew(model=None, quantiles=quantiles_forest,
                                                                          params=params)
        if parameters.get('algorithm') == 'QuantileKNN':
            quantile_estimator = helper.QuantileKNN(model=None, quantiles=quantiles_forest, params=params)

        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        lower, upper = helper.run_icp(nc, train, ytrain, test, idx_train, idx_cal, alpha)

        lower = lower * s + m
        upper = upper * s + m
        ytest = ytest * s + m

        header = ['QCP_lower', 'QCP_upper', 'NetPosUsd', 'prediction']
        size = upper / 2 + lower / 2
        table = np.vstack([lower, upper, ytest, size]).T

        dfncp = pd.DataFrame(table, columns=header)

        if i == 0:
            dfncp.to_csv(
                'QCP' + parameters.get('algorithm') + '_' + str(
                    np.round(parameters.get('alpha_')).astype(int)) + '_' + str(validation) + '.csv',
                encoding='utf-8', index=False)
        else:
            dfncp.to_csv(
                'QCP' + parameters.get('algorithm') + '_' + str(
                    np.round(parameters.get('alpha_')).astype(int)) + '_' + str(validation) + '.csv', mode='a',
                header=False, index=False)


def all_pvalues_algo(algorithm, validation, hypertuning=True):
    if hypertuning:
        df = pd.read_csv(algorithm + '_cv.csv')
        CP_dicts = df.loc[df.groupby('alpha_').CP_loss.idxmin()]
        CP_dicts = CP_dicts.drop(['CP_loss', 'NCP_loss'], axis=1).to_dict(orient='records')


    else:
        CP_dicts = dict()
        if algorithm == 'QuantileRegression':
            CP_dicts = {}

        elif algorithm == 'QuantileRandomForest':

            CP_dicts = {}
        elif algorithm == 'QuantileLightGBM':
            CP_dicts = {'min_samples_leaf': 5}

        elif algorithm == 'QuantileGradientBoosting':
            CP_dicts = {"min_samples_leaf": 5}
        else:
            CP_dicts = {}
        out = []

        for a in np.linspace(5, 95, 19):
            d = CP_dicts.copy()
            d['alpha_'] = np.round(a, 2)
            out.append(d)
        CP_dicts = out

    for dictio in tqdm(CP_dicts):
        dictio['algorithm'] = algorithm
        dictio['validation'] = validation

        train_and_test_quantile_QCP(dictio)

