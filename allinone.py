import datetime
import os
import random

import numpy as np
import pandas as pd
import torch


from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()


def interdays(date1, date2): #计算日期间隔 日为单位
    import time
    date1 = time.strptime(date1, "%Y-%m-%d")
    date2 = time.strptime(date2, "%Y-%m-%d")

    date1 = datetime.datetime(date1[0], date1[1], date1[2])
    date2 = datetime.datetime(date2[0], date2[1], date2[2])
    delta = date2-date1
    delta = int(delta.days)
    return delta

def setvalue(df):
    col = ['OILPRESSURE', 'CASINGPRESSURE',
           'BACKPRESSURE', 'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE',
           'PUMPINLETTEMPERTURE', 'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE',
           'FREQUENCY_POWER', 'CREEPAGE', 'ChokeDiameter', 'VIB', 'MOTORPOWER',
           'OIL(bbl)', 'GAS(103Mscf)', 'WATER(bbl)', 'LIQUID(bbl)',
           'GOR(Mscf/bbl)']
    maxvalue=[673.18429,729.5871669,421.8816811
              ,2833.664449	,5470.323583
              ,142.7808807	,162.2450233	,86	,2739.583333	,
              67.72916613	,17.40408222	,6400	,4.042011439	,
              269.0038819	,2770.631174	,1268607,	8816.409618,
              9887.255094	,2520.225308]
    minvalue=[-11.51908093,	-79.30167151,	24.76471922,	-200,
              -632.0488563,	-10	,-93.96819203,	0,	0,	0	,-17.6,	0	,
              -1.027523889	,0	,-9.96E-09	,0,	-5.18E-08,	0,	0]
    for i in range(len(col)):
        # df[col[i]].clip_upper(maxvalue[i])
        # df[col[i]].clip_lower(minvalue[i])
        df[col[i]].clip(upper=maxvalue[i],lower=minvalue[i],inplace=True)
    return df
def add_runtime(df):
    # df = pd.read_csv(dir)
    # print(df.columns)
    df1 = df.TIME
    df2 = df1.copy()
    df2[0] = 1
    for i in range(1, df1.shape[0] - 1):
        df2[i] = interdays(df1[i], df1[i + 1])
    df2.iloc[-1] = 1
    k = df2.copy()
    k[0] = 1
    for i in range(1, df2.shape[0]):
        k[i] = k[i - 1] + df2[i]
    df.drop("TIME", axis=1, inplace=True)
    df.insert(0, "RUNTIME", k)
    max=df["RUNTIME"].max()
    df["RUL"] = max - df["RUNTIME"]
    return df
def load_df(df):
    # print(df.shape)
    # print(df)
    # print(df.columns)
    # df.col = ['id', 'RUNTIME', 'CASINGPRESSURE', 'PUMPINLETPRESSURE',
    #        'PUMPOUTPRESSURE', 'OILPRESSURE', 'PUMPINLETTEMPERTURE', 'CURRENTS',
    #        'CREEPAGE', 'MOTORPOWER', 'MOTORTEMPERTURE', 'VIB', 'VOLTAGE',
    #        'ChokeDiameter', 'FREQUENCY_POWER', 'RUL']

    # 先按照'id'列的元素进行排序，当'id'列的元素相同时按照'cycle'列进行排序
    # df = df.sort_values(['id', 'RUNTIME'])
    # rul = pd.DataFrame(df.groupby('id')['RUNTIME'].max()).reset_index()
    # rul.columns = ['id', 'max']
    # # 将rul通过'id'合并到train_df上，即在相同'id'时将rul里的max值附在train_df的最后一列
    # df = df.merge(rul, on=['id'], how='left')
    # # 加一列，列名为'RUL'
    # df[''] = df['max'] - df['runtime']
    # # 将'max'这一列从train_df中去掉
    # df.drop('max', axis=1, inplace=True)
    # """MinMax normalization train"""
    # 将'cycle'这一列复制给新的一列'cycle_norm'
    df['RUNTIME_NORM'] = df['RUNTIME']
    # 在列名里面去掉'id', 'cycle', 'RUL'这三个列名
    col_norm = df.columns.difference([ 'RUNTIME', 'RUL'])
    # 对剩下名字的每一列分别进行特征放缩

    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[col_norm]),
                           columns=col_norm,
                           index=df.index)
    # 将之前去掉的再加回特征放缩后的列表里面
    join_df = df[df.columns.difference(col_norm)].join(norm_df)
    # 恢复原来的索引
    df = join_df.reindex(columns=df.columns)
    df.loc[df["RUL"] >= 1300, "RUL"] = 1300
    return df
def time2day(df):

    # df = pd.read_csv(dir)
    df1 = df["THETIME"].copy()
    for i, time in enumerate(df1):
        df1[i] = time.split(sep=" ")[0]
    df["TIME"] = df1
    df.drop(columns=["THETIME"], inplace=True)
    day = df.groupby(df.TIME).mean()
    day["TIME"] = day.index
    day.fillna(day.mean(),inplace=True)

    return day
def df_process(df):
    df=setvalue(df)
    df=time2day(df)
    df=add_runtime(df)
    df=load_df(df)
    return df

def df_totensor(df,seq_cols):
    x= df[seq_cols].values
    x.dtype='float64'
    y=df["RUL"].values.tolist()
    y=np.array(y,dtype='float64')

    return torch.from_numpy(x),torch.from_numpy(y)



if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(20)
    torch.set_default_tensor_type(torch.DoubleTensor)
    df1=pd.read_csv("PL19-3-A01H2实时数据 2019-07-03-2021-06-22.csv")
    df=df_process(df1)
    seq_col = ['OILPRESSURE', 'CASINGPRESSURE', 'BACKPRESSURE',
               'PUMPINLETPRESSURE', 'PUMPOUTPRESSURE', 'PUMPINLETTEMPERTURE',
               'MOTORTEMPERTURE', 'CURRENTS', 'VOLTAGE', 'FREQUENCY_POWER', 'CREEPAGE',
               'ChokeDiameter', 'VIB', 'MOTORPOWER', 'OIL(bbl)', 'GAS(103Mscf)',
               'WATER(bbl)', 'LIQUID(bbl)', 'GOR(Mscf/bbl)', 'RUNTIME_NORM']
    x,_=df_totensor(df,seq_col)
    x=x.view(1,-1,20)

    net=torch.load("goodnet.pt", map_location='cpu')
    net=net.double()
    rul=net(x.double()).detach().numpy().item()
    print('当前剩余寿命为:%s天'%(rul))