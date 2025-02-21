# @File : Test_Mpl_attention_model.py
# @Time : 2024/4/27 10:17
# @Author : Qiao Hanyang
# @Motto : 凡所有相，皆是虚妄。若见诸相非相，则见如来。——《金刚经》
# @Phone : 18059898389
# @Address : XMU

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
import warnings
import pandas as pd
import pandas as pd
import joblib
import torch.nn as nn
from data_process import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import math
def MAE(pred, target):
    return torch.sum(torch.abs(pred - target)) / len(pred)


def MSE(pred, target):
    return torch.sum((pred - target) ** 2) / len(pred)


def RMSE(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2) / len(pred))


# def MAPE(pred, target):
#     return torch.sum(torch.abs((target - pred) / (target))) * (100 / len(pred))
def MAPE(pred, target):
    epsilon = 8e-1
    return torch.sum(torch.abs((target - pred) / (target + epsilon))) * (100 / len(pred))

def R2(pred, target):
    return 1 - (torch.sum((target - pred) ** 2)) / (torch.sum((target - torch.mean(target)) ** 2))
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_ld(df_total, batch_size):
    x = torch.tensor(np.array(df_total.iloc[:,:-1]), dtype=torch.float)
    y = torch.tensor(np.array(df_total.iloc[:,-1:]), dtype=torch.float)
    print("dasdkasfhaklshfkh", x.shape, y.shape)
    torch_dataset = TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    return loader
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
import os.path as osp


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads):
        super(SpatialAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(heads)
        ])

    def forward(self, latitude_longitude):
        # 分离经度和纬度
        latitude, longitude = latitude_longitude[:, 0], latitude_longitude[:, 1]

        # 更复杂的特征转换
        transformed_lat_long = torch.cat((
            torch.sin(latitude.unsqueeze(-1)),
            torch.cos(latitude.unsqueeze(-1)),
            torch.sin(longitude.unsqueeze(-1)),
            torch.cos(longitude.unsqueeze(-1)),
            # 可以加入更多的特征转换
        ), dim=1)

        # MLP处理
        transformed_features = self.mlp(transformed_lat_long)

        # 多头注意力
        attention_weights = [torch.sigmoid(head(transformed_features)) for head in self.attention_heads]
        attention_weights = torch.cat(attention_weights, dim=1)

        # 返回每个头的注意力权重作为一个单独的数值
        return attention_weights


input_dim = 4  # 经纬度数据转换后的维度
hidden_dim = 64  # 隐藏层维度，可以根据实验需要调整
heads = 2  # 注意力头的数量，可以根据实验需要调整

# 创建 SpatialAttention 实例
spatial_attention = SpatialAttention(input_dim, hidden_dim, heads)


class Net(nn.Module):
    def __init__(self, in_channels, hid_channels1, hid_channels2, hid_channels3, out_channels):
        super(Net, self).__init__()
        # 假设 SpatialAttention 输出的维度与 geo_data 维度相同

        self.attention_weights = spatial_attention
        # 创建 SpatialAttention 实例

        # 输入层现在接收经过注意力机制处理的地理数据和其他数据的组合
        self.input = nn.Linear(in_channels + 2, hid_channels1)

        self.line1 = nn.Sequential(
            nn.Linear(hid_channels1, hid_channels2),
            nn.ReLU(),
            nn.BatchNorm1d(hid_channels2),
            nn.Dropout(p=0.5)
        )
        self.line2 = nn.Sequential(
            nn.Linear(hid_channels2, hid_channels3),
            nn.ReLU(),
            nn.BatchNorm1d(hid_channels3),
            nn.Dropout(p=0.3)
        )
        self.line3 = nn.Sequential(
            nn.Linear(hid_channels3, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        geo_data = x[:, -2:]  # 提取经度和纬度
        other_data = x[:, :-2]  # 提取除经纬度外的其他特征

        geo_encoded = self.attention_weights(geo_data)
        combined_data = torch.cat([other_data, geo_encoded], dim=1)  # 在输入层之前拼接

        x1 = self.input(combined_data)
        x2 = self.line1(x1)
        x3 = self.line2(x2)
        x4 = self.line3(x3)

        return x4


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    folder_path =   'F:\\HiSea2\\matchdata\\风雨\\'
    net = torch.load('xxx.pth')
    # 获取文件夹中所有的 .npy 文件
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    # 初始化一个空的列表来存储所有转换后的二维数组
    arrays_2d = []

    # 遍历每一个 .npy 文件
    for npy_file in npy_files:
        file_path = os.path.join(folder_path, npy_file)
        data = np.load(file_path)
        arr_2d = data.reshape(36, -1)
        arr_2d_transposed = arr_2d.transpose()
        arrays_2d.append(arr_2d_transposed)

    # 使用 np.concatenate 函数合并所有的二维数组
    final_array = np.concatenate(arrays_2d, axis=0)

    # 将前12列作为特征值，最后一列作为标签
    features = final_array[:, :12]
    labels = final_array[:, -9]
    F0_values = final_array[:, 19:27]
    era5data = final_array[:, 12:18]

    lat = final_array[:, -4]
    lon = final_array[:, -3]
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    lon_cos = np.cos(lon_rad)
    lon_sin = np.sin(lon_rad)
    lat_cos = np.cos(lat_rad)
    lat_sin = np.sin(lat_rad)

    # 寻找含有 NaN 的行
    nan_rows = np.isnan(features).any(axis=1) | np.isnan(labels)

    # 只保留没有 NaN 的行
    valid_features = features[~nan_rows]
    valid_labels = labels[~nan_rows]
    valid_F0_values = F0_values[~nan_rows]
    valid_era5 = era5data[~nan_rows]
    valid_lon_cos = lon_cos[~nan_rows]
    valid_lon_sin = lon_sin[~nan_rows]
    valid_lat_cos = lat_cos[~nan_rows]
    valid_lat_sin = lat_sin[~nan_rows]
    valid_lat = lat[~nan_rows]
    valid_lon = lon[~nan_rows]

    # 先将角度转换为余弦值
    valid_features[:, 0:4] = np.cos(np.radians(valid_features[:, 0:4]))

    # num_bands = 8
    #
    # # 初始化一个新的数组来存储归一化后的特征，以防改变原始数据
    # normalized_features = np.zeros_like(valid_features[:, 4:12])
    #
    # # 循环每个波段进行归一化
    # for i in range(num_bands):
    #     band_index = 4 + i  # 计算当前波段在 valid_features 中的列索引
    #     # 对每个样本进行归一化处理
    #     normalized_features[:, i] = valid_features[:, band_index] / (valid_F0_values[:, i] * valid_features[:,0])
    #
    # # 现在 normalized_features 存储了归一化后的波段数据
    # # 如果需要，可以将这些值替换回原始数组
    # valid_features[:, 4:12] = normalized_features

    valid_labels_2d = valid_labels[:, np.newaxis]
    rows_with_greater_than_9 = np.any(valid_labels_2d > 100.0, axis=1)

    valid_features = valid_features[~rows_with_greater_than_9]
    valid_era5 = valid_era5[~rows_with_greater_than_9]
    valid_labels = valid_labels[~rows_with_greater_than_9]
    valid_lon_cos = valid_lon_cos[~rows_with_greater_than_9]
    valid_lon_sin = valid_lon_sin[~rows_with_greater_than_9]
    valid_lat_cos = valid_lat_cos[~rows_with_greater_than_9]
    valid_lat_sin = valid_lat_sin[~rows_with_greater_than_9]
    valid_lat = valid_lat[~rows_with_greater_than_9]
    valid_lon = valid_lon[~rows_with_greater_than_9]
    valid_labels = valid_labels[:, np.newaxis]

    # valid_labels = np.log10(valid_labels)

    # 合并特征和其他数据
    box = np.hstack((valid_features, valid_era5))

    wind1 = box[:, 12];
    wind2 = box[:, 13];
    wind = np.sqrt(wind1 ** 2 + wind2 ** 2)
    wind = wind[:, np.newaxis]
    box_ear5 = np.hstack((wind, valid_era5[:, 2:]))

    box = np.hstack((valid_features, box_ear5, valid_lon[:, np.newaxis], valid_lat[:, np.newaxis], valid_labels))
    print(box.shape)

    condition = (box[:, 4:12] < 200.0) | (box[:, 4:12] > 2500.0)

    # 使用any(axis=1)来检查每一行是否有任何一个数据符合条件
    condition = condition.any(axis=1)

    # 过滤掉不符合条件的行
    filtered_box = box[~condition]
    # filtered_box = pd.DataFrame(filtered_box)

    import joblib
    robust_scaler = joblib.load('xxx.pkl')
    some_thing = filtered_box[:, :17]
    features_scaled = robust_scaler.transform(some_thing)
    xz = filtered_box[:, 17:19]
    final_features = np.hstack((features_scaled,xz,filtered_box[:,-1][:, np.newaxis]))
    train_loader = pd.DataFrame(final_features)



    x = torch.tensor(np.array(train_loader.iloc[:,:-1]), dtype=torch.float)
    y = torch.tensor(np.array(train_loader.iloc[:,-1:]), dtype=torch.float)
    print(x.shape,y.shape)
    batch_x = x; batch_y = y
    epoch_loss_test = []
    epoch_loss_mae_test = []
    epoch_loss_rmse_test = []
    epoch_loss_r2_test = []
    epoch_loss_mape_test = []


    print(batch_y.shape,batch_x.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)


    out = net(batch_x.to(device))

    print(out.shape)
    print(batch_y.shape)
    print(MAE(out, batch_y.to(device)))
    print(MSE(out, batch_y.to(device)))
    print(MAPE(out, batch_y.to(device)))
    print(R2(out, batch_y.to(device)))

    out = out.detach().cpu().numpy();batch_y = batch_y.detach().cpu().numpy()
from scipy import stats
"""可视化"""
from itertools import chain
from scipy.stats import gaussian_kde

l1 = list(out)
onid1 = list(chain.from_iterable(l1))
l2 = list(batch_y)
onid2 = list(chain.from_iterable(l2))
sample = list(range(0, len(onid2), 100))
x = np.array(onid2)[sample]
y = np.array(onid1)[sample]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use('seaborn-darkgrid')

# 假设数据x, y已经定义
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

density_threshold = 0.0008
mask = z > density_threshold
x, y, z = x[mask], y[mask], z[mask]

fig, ax = plt.subplots(figsize=(10, 12))  # 留出更多空间给颜色棒
ax.set_aspect('equal')

sc = ax.scatter(x, y, c=z, s=10, cmap='viridis', edgecolor='w', linewidth=0.01)

shared_bins = np.histogram_bin_edges(x, bins=128)

divider = make_axes_locatable(ax)
ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histy.yaxis.set_tick_params(labelleft=False)

ax_histx.hist(x, bins=shared_bins, density=True, alpha=0.75, color='orange')
ax_histy.hist(y, bins=shared_bins, orientation='horizontal', density=True, alpha=0.75, color='orange')

ax.set_xlabel('S3-Chla (mg.m-3)', fontsize=18)
ax.set_ylabel('HS2-Chla (mg.m-3)', fontsize=18)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.plot([np.min([x, y]), np.max([x, y])], [np.min([x, y]), np.max([x, y])], 'k--', linewidth=2)

# 添加颜色栏在底部，调整位置和尺寸
axins = inset_axes(ax, width="100%", height="5%", loc='lower center',
                   bbox_to_anchor=(0, -0.16, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cbar = fig.colorbar(sc, cax=axins, orientation='horizontal')
cbar.set_label('Density', rotation=0, labelpad=10, fontsize=14)
# 设置颜色棒的刻度
cbar.set_ticks([np.min(z), np.mean(z), np.max(z)])  # 设置最小值，平均值，最大值为刻度
cbar.set_ticklabels([f'{np.min(z):.2f}', f'{np.mean(z):.2f}', f'{np.max(z):.2f}'])  # 标记刻度标签
cbar.ax.tick_params(labelsize=15)

# 设置文本字符串，包括一个百分比
N = 48127
r2 = 0.77
mape = 22.23  # 作为百分比表示
mae = 0.57

# 格式化文本字符串
text_str = f'N: {N}\n$R^2$: {r2:.2f}\nMAPE: {mape:.2f}%\nMAE: {mae:.2f}'

# 将文本添加到图表
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=16,
         verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

plt.show()
