# -*- coding: utf-8 -*-
# @Time : 2022/7/3 18:06
# @Author : Qiaohy
# @FileName: data_process.py
# @Email : 2279015365@qq.com
# @Software: PyCharm
import os
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from torch.utils.data.dataset import TensorDataset
import torch.utils.data as Data
import torch
from sklearn.model_selection import KFold

def MAE(pred, target):
    return torch.sum(torch.abs(pred - target)) / len(pred)


def MSE(pred, target):
    return torch.sum((pred - target) ** 2) / len(pred)


def RMSE(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2) / len(pred))


def MAPE(pred, target):
    return torch.sum(torch.abs((target - pred) / (target))) * (100 / len(pred))


def R2(pred, target):
    return 1 - (torch.sum((target - pred) ** 2)) / (torch.sum((target - torch.mean(target)) ** 2))

# 将目录下的数据集拼接
def get_raw_data(path):
    cnt = 0
    df_total = pd.DataFrame([])
    for file in os.listdir(path):
        if ('.csv' in file or '.CSV' in file) and 'raw' not in file and os.path.isfile(os.path.join(path, file)):
            cnt += 1
            df = pd.read_csv(os.path.join(path, file), header=None, dtype=float)
            from sklearn.utils import shuffle
            df = shuffle(df, n_samples=len(df))
            SOZ = df.iloc[:,8]
            cos_SOZ = np.cos(SOZ / 180 * np.pi)
            # fuzhou 0301
            a1 = 1000*((df.iloc[:, 0] - 1675.0705 - df.iloc[:, 7] ) * 0.001753 + 0.058615)/ (187 * cos_SOZ)
            a2 = 1000*((df.iloc[:, 1] - 1437.0512- df.iloc[:, 7]) * 0.001436 + 0.042528)/(192 * cos_SOZ)
            a3 = 1000*((df.iloc[:, 2] - 1776.93- df.iloc[:, 7]) * 0.0013089 + 0.0403429)/(178 * cos_SOZ)
            a4 = 1000*((df.iloc[:, 3] - 1805.1858- df.iloc[:, 7]) * 0.0012524 + 0.042795)/(162 * cos_SOZ)
            a5 = 1000*((df.iloc[:, 4] - 2041.09615- df.iloc[:, 7]) * 0.0011617 + 0.07226)/(147 * cos_SOZ)
            a6 = 1000*((df.iloc[:, 5] - 1763.551- df.iloc[:, 7]) * 0.000914 + 0.06627)/(140 * cos_SOZ)
            a7 = 1000*((df.iloc[:, 6] - 1696.12820- df.iloc[:, 7]) * 0.001988981 + 0.107663544)/(128 * cos_SOZ)
            # a8 = 1000*((df.iloc[:, 7] - 1280.923- df.iloc[:, 7]) * 0.000889854 + 0.083192440)/(95 * cos_SOZ)

            w1 = df.iloc[:, 7];w2 = df.iloc[:, 8];w3 = df.iloc[:, 9];w4 = df.iloc[:, 10]
            y1 = df.iloc[:,11]

            # df = df.sample(frac=1.0)

            df_total_l = pd.concat([df_total, a1,a2,a3,a4,a5,a6,a7], axis=1)
            df_angle = pd.concat([w1,w2,w3,w4,y1], axis=1)

            # df_angle[df_angle.columns[0:-1]] = (df_angle[df_angle.columns[0:-1]] - df_angle[
            #     df_angle.columns[0:-1]].min()) / (   df_angle[df_angle.columns[0:-1]].max() - df_angle[df_angle.columns[0:-1]].min())
            df_total = pd.concat([df_total_l,df_angle],axis =1)
            # print(df_total)
            print("success")
    cols = ['attr' + str(i + 1) for i in range(df_total.shape[1] - 1)]
    cols.append('label')
    df_total.columns = cols
    # df_total[df_total.columns[0:-1]] = (df_total[df_total.columns[0:-1]] - df_total[df_total.columns[0:-1]].min()) / (
    #         df_total[df_total.columns[0:-1]].max() - df_total[df_total.columns[0:-1]].min())
    print(f'共{cnt}个子数据集')
    df = df.dropna()
    print(f'拼接后数据量{df_total.shape[0]}')
    return df_total



# 剔除异常点
def ex_dot(df):
    raw_int = df.shape[0]
    model = IsolationForest()
    model.fit(df)
    res = model.predict(df)

    df = df[res == 1]
    new_int = df.shape[0]
    df = df.sample(frac=1.0)
    print(f'去除异常点{raw_int - new_int}个')
    return df

# 交叉验证K=任意


# 根据df构造数据流
def construct_ld(df_total, batch_size):
    x = torch.tensor(np.array(df_total.iloc[:,:-1]), dtype=torch.float)
    y = torch.tensor(np.array(df_total.iloc[:,-1:]), dtype=torch.float)
    print("dasdkasfhaklshfkh", x.shape, y.shape)
    print(y)
    torch_dataset = TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader

