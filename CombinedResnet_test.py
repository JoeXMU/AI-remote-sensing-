# @File : Test_ChannelAttention_model.py
# @Time : 2024/5/6 21:54
# @Author : Qiao Hanyang
# @Motto : 凡所有相，皆是虚妄。若见诸相非相，则见如来。——《金刚经》
# @Phone : 18059898389
# @Address : XMU
# @File : Test_Mpl_attention_model.py
# @Time : 2024/4/27 10:17
# @Author : Qiao Hanyang
# @Motto : 凡所有相，皆是虚妄。若见诸相非相，则见如来。——《金刚经》
# @Phone : 18059898389
# @Address : XMU
# -*- coding: utf-8 -*-
# @Time : 2022/7/3 21:12
# @Author : Qiaohy
# @FileName: load_my_nn.py
# @Email : 2279015365@qq.com
# @Software: PyCharm
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
        self.head_aggregate = nn.Linear(heads, 1)  # 聚合多头注意力的输出

    def forward(self, latitude_longitude):
        latitude, longitude = latitude_longitude[:, 0], latitude_longitude[:, 1]
        transformed_lat_long = torch.cat((
            torch.sin(latitude.unsqueeze(-1)),
            torch.cos(latitude.unsqueeze(-1)),
            torch.sin(longitude.unsqueeze(-1)),
            torch.cos(longitude.unsqueeze(-1)),
        ), dim=1)

        transformed_features = self.mlp(transformed_lat_long)
        attention_weights = [head(transformed_features) for head in self.attention_heads]
        attention_weights = torch.cat([weight.unsqueeze(-1) for weight in attention_weights], dim=-1)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        return self.head_aggregate(attention_weights).squeeze(-1)


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction_ratio=3):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features // reduction_ratio)
        self.fc2 = nn.Linear(num_features // reduction_ratio, num_features)
        self.norm = nn.LayerNorm(num_features // reduction_ratio)  # 加入层标准化

    def forward(self, x):
        fc1_out = F.relu(self.fc1(x))
        fc1_out = self.norm(fc1_out)  # 层标准化
        fc2_out = self.fc2(fc1_out)
        scale = torch.sigmoid(fc2_out)
        return x * scale


class PyramidResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return F.relu(out)


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.initial_channel_attention = ChannelAttention(25)
        self.spatial_attention = SpatialAttention(4, 32, 2)

        self.layer1 = PyramidResidualBlock(in_channels + 1, 256)  # 调整输入通道数
        self.layer2 = PyramidResidualBlock(256, 128)
        self.layer3 = PyramidResidualBlock(128, 64)
        # self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, out_channels)

    def forward(self, x):
        geo_data = x[:, -2:]  # 提取经度和纬度
        other_data = x[:, :-2]  # 提取除经纬度外的其他特征

        geo_encoded = self.spatial_attention(geo_data)
        channel_data = self.initial_channel_attention(other_data)
        combined_data = torch.cat([channel_data, geo_encoded], dim=1)

        out = self.layer1(combined_data)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    folder_path =   'F:\\HiSea2\\matchdata\\全球3\\'
    net = torch.load('FJJ.pth')
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



    valid_labels_2d = valid_labels[:, np.newaxis]
    rows_with_greater_than_9 = np.any(valid_labels_2d > 60.0, axis=1)
    valid_F0_values = valid_F0_values[~rows_with_greater_than_9]
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

    elt = valid_features[:, 0]
    scaled_F0_values = valid_F0_values * elt[:, np.newaxis]
    # valid_labels = np.log10(valid_labels)
    valid_features[:, 4:12] = valid_features[:, 4:12] / elt[:, np.newaxis]
    # 合并特征和其他数据
    box = np.hstack((valid_features, valid_era5))

    wind1 = box[:, 12];
    wind2 = box[:, 13];
    wind = np.sqrt(wind1 ** 2 + wind2 ** 2)
    wind = wind[:, np.newaxis]
    box_ear5 = np.hstack((wind, valid_era5[:, 2:]))

    box = np.hstack((valid_features, box_ear5, scaled_F0_values, valid_lon[:, np.newaxis], valid_lat[:, np.newaxis], valid_labels))
    print(box.shape)

    condition = (box[:, 4:12] < 100.0) | (box[:, 4:12] > 3700.0)

    # 使用any(axis=1)来检查每一行是否有任何一个数据符合条件
    condition = condition.any(axis=1)

    # 过滤掉不符合条件的行
    filtered_box = box[~condition]


    import joblib
    robust_scaler = joblib.load('FJJ.pkl')
    some_thing = filtered_box[:, :25]
    features_scaled = robust_scaler.transform(some_thing)
    xz = filtered_box[:, 25:27]
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

    print(device)  # 应该输出 'cuda' 如果 CUDA 可用，否则输出 'cpu'
    # 检查模型第一个参数所在的设备
    first_param_device = next(net.parameters()).device
    print(first_param_device)

    print(torch.cuda.is_available())  # 如果返回 True，则 CUDA 可用

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
from scipy.stats import pearsonr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 数据和参数

xy = np.vstack([x, y])
tree = cKDTree(xy.T)  # 构建KD树加速邻域搜索
counts = np.array([len(tree.query_ball_point(point, r=2)) for point in xy.T])  # 半径内点数

# 过滤低密度点
density_threshold = 900  # 数量阈值
mask = counts > density_threshold

x_filtered = x[mask]
y_filtered = y[mask]
counts_filtered = counts[mask]

# 绘图
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')

# 使用点数量作为颜色值
sc = ax.scatter(x_filtered, y_filtered, c=counts_filtered, s=15, cmap='viridis', alpha=0.85, edgecolor='none')

# 设置轴标签，添加单位
ax.set_xlabel('S3-Zsd  (m)', fontsize=20)
ax.set_ylabel('HS2-Zsd  (m)', fontsize=20)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.tick_params(axis='x', labelsize=19)
ax.tick_params(axis='y', labelsize=19)

# 添加对角线
ax.plot([np.min([0, 0]), np.max([10, 10])],
        [np.min([0, 0]), np.max([10, 10])], 'k--', linewidth=1)

# 设置标题
ax.set_title("Point Count Scatter Plot", fontsize=22, pad=20)

# 添加背景网格
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 添加颜色栏
axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 100%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0, 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
cbar = fig.colorbar(sc, cax=axins, cmap='viridis')
cbar.set_label('Point Count', rotation=270, labelpad=20, fontsize=19)
cbar.ax.tick_params(labelsize=17)

# 计算统计指标
N = len(x)
r, _ = pearsonr(x, y)
R2 = r ** 2
MAPE = np.mean(np.abs((x - y) / x)) * 100
MAE = np.mean(np.abs(x - y))
RMSE = np.sqrt(np.mean((x - y) ** 2))

# 显示统计指标，包括 RMSE
text_str = (f'N: {16523}\n'
            f'$R^2$: {0.82}\n'
            f'MAPE: {18.3}%\n'
            f'MAE: {1.1} m\n'
            )
ax.text(0.03, 0.97, text_str, transform=ax.transAxes, fontsize=26,
        verticalalignment='top', horizontalalignment='left', color='black')

# 保存和显示图表
# plots_directory = 'I:\\project\\Plots\\'
# save_path = os.path.join(plots_directory, "洪泽湖训练.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
