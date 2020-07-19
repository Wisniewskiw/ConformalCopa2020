import scipy.stats
from tqdm import tqdm
def get__window_ma(df,list,alpha):


    normalq = scipy.stats.norm.ppf(1 - alpha / 2)
    for window in tqdm(list):
        df['m'] = df['NetPosUsd'].shift(1).rolling(window, min_periods=1).mean()
        df['std'] = df['NetPosUsd'].shift(1).rolling(window, min_periods=1).std()
        df['lower_benchmark'] = DQR.df['m'] - normalq * DQR.df['std']
        df['upper_benchmark'] = DQR.df['m'] + normalq * DQR.df['std']
        df=df['QdfTime','m','lower_benchmark','upper_benchmark']
        df.to_csv('MvgAvgWindow'+str(window)+'.csv')

def get__window_alpha(df,window):

    normalq = scipy.stats.norm.ppf(1 - alpha / 2)
    for alpha in tqdm(np.linspace(0.05,0.95,19)):
        normalq = scipy.stats.norm.ppf(1 - alpha / 2)
        df['m'] = df['NetPosUsd'].shift(1).rolling(window, min_periods=1).mean()
        df['std'] = df['NetPosUsd'].shift(1).rolling(window, min_periods=1).std()
        df['lower_benchmark'] = DQR.df['m'] - normalq * DQR.df['std']
        df['upper_benchmark'] = DQR.df['m'] + normalq * DQR.df['std']
        df=df['QdfTime','m','lower_benchmark','upper_benchmark']
        df.to_csv('MvgAvgalpha'+str(int(100*alpha))+'.csv')
