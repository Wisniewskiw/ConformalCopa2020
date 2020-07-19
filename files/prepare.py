import pandas as pd
import datetime
from tqdm import tqdm
def addlags(df):

    df['weekday'] = df.QdfTime.dt.weekday
    df['hour'] = df.QdfTime.dt.hour
    for i in range(1, 25):
        df["lag_{}".format(i)] = df.NetPosUsd.shift(i)
    return df

def market_stats(df):
    import datetime
    #one has to check those dates
    summer_start_2014=datetime.datetime(2014, 3, 26, 1, 0)
    summer_end_2014=datetime.datetime(2014, 10, 29, 2, 0)
    summer_start_2015=datetime.datetime(2015, 3, 25, 1, 0)
    summer_end_2015=datetime.datetime(2015, 10, 28, 2, 0)
    summer_start_2016=datetime.datetime(2016, 3, 31, 1, 0)
    summer_end_2016=datetime.datetime(2016, 10, 27, 2, 0)
    Sydney_winter_hours=[1,2,3,4,5,6,21,22,23,24]
    Tokyo_winter_hours=[1,2,3,4,5,6,7,8,23,24]
    London_winter_hours=[8,9,10,11,12,13,14,15,16,17]
    NewYork_winter_hours=[13,14,15,16,17,18,19,20,21,22]
    Sydney_summer_hours=[1,2,3,4,5,6,7,22,23,24]
    Tokyo_summer_hours=[1,2,3,4,5,6,7,8,23,24]
    London_summer_hours=[7,8,9,10,11,12,13,14,15,16]
    NewYork_summer_hours=[12,13,14,15,16,17,18,19,20,21]

    winter=(df.QdfTime<summer_start_2014) |((df.QdfTime>summer_end_2014)& (df.QdfTime<summer_start_2015)) |((df.QdfTime>summer_end_2015)& (df.QdfTime<summer_start_2016))  |(df.QdfTime>summer_end_2016)
    summer=~winter
    df['Sydney_open']=winter*(df.QdfTime.dt.hour.isin(Sydney_winter_hours))+summer*(df.QdfTime.dt.hour.isin(Sydney_summer_hours))
    df['Tokyo_open']=winter*(df.QdfTime.dt.hour.isin(Tokyo_winter_hours))+summer*(df.QdfTime.dt.hour.isin(Tokyo_summer_hours))
    df['London_open']=winter*(df.QdfTime.dt.hour.isin(London_winter_hours))+summer*(df.QdfTime.dt.hour.isin(London_summer_hours))
    df['NewYork_open']=winter*(df.QdfTime.dt.hour.isin(NewYork_winter_hours))+summer*(df.QdfTime.dt.hour.isin(NewYork_summer_hours))
    df['HowManyOpen']=df['Sydney_open']+df['Tokyo_open']+df['London_open']+df['NewYork_open']

    for market in ['Sydney','Tokyo','London','NewYork']:
        df[market+'minutes_to_close']=df[::-1].groupby((df[market+'_open'][::-1] != 1).cumsum()).cumcount()[::-1]
        df[market+'minutes_to_open']=df[::-1].groupby((df[market+'_open'][::-1] != 0).cumsum()).cumcount()[::-1]
        df[market+'minutes_after_opening']=df.groupby((df[market+'_open'] != 1).cumsum()).cumcount()

    return df

def five_day_split(df):
    for i in tqdm(range(29)):
        x = df[i * (24 * 5):df.shape[0] - 29 * 24 * 5 + (i + 1) * (24 * 5)]

        x.to_csv('data\EURUSD_NETPOSUSD_hourly_regression_split' + str(i) + '.csv')