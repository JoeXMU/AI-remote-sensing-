# @File : CombinedResnet.py
# @Time : 2024/5/9 10:33
# @Author : Qiao Hanyang
# @Motto : 凡所有相，皆是虚妄。若见诸相非相，则见如来。——《金刚经》
# @Phone : 18059898389
# @Address : XMU

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import torch.nn.functional as F
from matplotlib.ticker import FormatStrFormatter
import os
from matplotlib.ticker import FormatStrFormatter
import torch.nn as nn
import torch.optim as optim
from data_process import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import math
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    folder_path = 'F:\\HiSea2\\matchdata\\全球2\\'

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
    rows_with_greater_than_9 = np.any(valid_labels_2d > 100.0, axis=1)

    valid_features = valid_features[~rows_with_greater_than_9]
    valid_F0_values = valid_F0_values[~rows_with_greater_than_9]
    valid_era5 = valid_era5[~rows_with_greater_than_9]
    valid_labels = valid_labels[~rows_with_greater_than_9]
    valid_lon_cos = valid_lon_cos[~rows_with_greater_than_9]
    valid_lon_sin = valid_lon_sin[~rows_with_greater_than_9]
    valid_lat_cos = valid_lat_cos[~rows_with_greater_than_9]
    valid_lat_sin = valid_lat_sin[~rows_with_greater_than_9]
    valid_lat = valid_lat[~rows_with_greater_than_9]
    valid_lon = valid_lon[~rows_with_greater_than_9]
    valid_labels = valid_labels[:, np.newaxis]

    elt = valid_features[:,0]
    scaled_F0_values = valid_F0_values * elt[:, np.newaxis]
    # valid_labels = np.log10(valid_labels)
    valid_features[:, 4:12] = valid_features[:, 4:12] / elt[:, np.newaxis]
    # 合并特征和其他数据
    box = np.hstack((valid_features, valid_era5))

    wind1 = box[:, 12]; wind2 = box[:,13] ;
    wind =  np.sqrt(wind1**2 + wind2**2)
    wind = wind[:, np.newaxis]
    box_ear5 = np.hstack((wind, valid_era5[:,2:]))

    box = np.hstack((valid_features,box_ear5, scaled_F0_values, valid_lon[:, np.newaxis],valid_lat[:, np.newaxis],valid_labels))



    condition = (box[:, 4:12] < 100.0) | (box[:, 4:12] > 2500.0)

    # 使用any(axis=1)来检查每一行是否有任何一个数据符合条件
    condition = condition.any(axis=1)

    # 过滤掉不符合条件的行
    filtered_box = box[~condition]
    filtered_box = pd.DataFrame(filtered_box)
    # 聚类以减少数据点数

    # scaler = StandardScaler()
    robust_scaler = RobustScaler()


    filtered_box.iloc[:, :25] = robust_scaler.fit_transform(filtered_box.iloc[:, :25])

    # 保存 RobustScaler 对象
    joblib.dump(robust_scaler, 'FJJ.pkl')


    import torch.nn as nn
    import torch
    import torch.utils.data as Data

    def construct_ld(df_total, batch_size):
        x = torch.tensor(np.array(df_total.iloc[:,:-1]), dtype=torch.float)
        y = torch.tensor(np.array(df_total.iloc[:,-1:]), dtype=torch.float)
        print("dasdkasfhaklshfkh", x.shape, y.shape)
        torch_dataset = TensorDataset(x, y)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        return loader


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
            fc1_out = self.norm(fc1_out)
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

            self.layer1 = PyramidResidualBlock(in_channels + 1, 128)  # 调整输入通道数
            self.layer2 = PyramidResidualBlock(128, 128)
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



    import warnings
    warnings.filterwarnings('ignore')




    import pandas as pd
    data_total = pd.DataFrame(filtered_box)
    print("wishinidie", data_total.shape)

    import numpy as np
    import torch
    import random
    import os


    def seed_everything(seed=42):
        random.seed(seed)  # Python内置的random库
        os.environ['PYTHONHASHSEED'] = str(seed)  # 环境变量
        np.random.seed(seed)  # NumPy库
        torch.manual_seed(seed)  # PyTorch
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False



    def get_loader(df_total, batch_size, *, rate=0.15):
        # 在这个函数调用前，确保已经设置了随机种子
        df_total = df_total.sample(frac=1.0)  # pandas的sample也受随机种子的影响
        cut = int(df_total.shape[0] * rate)
        df_train = df_total[:cut]
        df_test = df_total[cut:]

        print(f'Train Dataset: {df_train.shape}')
        print(f'Test Dataset: {df_test.shape}')

        # 假设construct_ld是用于构建数据加载器的函数
        return construct_ld(df_train, batch_size), construct_ld(df_test, 8000000000)


    # 设置随机种子
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    """构建数据流"""
    batch_size = 300

    train_loader, test_loader = get_loader(data_total, batch_size)
    """定义超参数"""
    in_channels,  out_channels  = 25,  1
    lr = 0.0006
    L2 = 0.05  # L2正则化防止过拟合
    # attention_size = 20
    net = Net(in_channels=in_channels, out_channels=out_channels).to(device)
    net = net.to(device)
    optim = torch.optim.NAdam(net.parameters(), lr=lr, weight_decay=L2)   # weight_decay=L2
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.88)

    loss_func = nn.MSELoss()
    epochs = 5
    train_loss = []
    test_loss = []

    """训练及测试"""
    for epoch in range(epochs):
        epoch_loss = []
        for batch_x, batch_y in train_loader:
            # print("size",batch_x.shape,batch_y.shape)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # print(batch_y.mean())
            optim.zero_grad()
            out = net(batch_x)
            batch_loss = loss_func(out, batch_y)
            batch_loss.backward()
            optim.step()
            epoch_loss.append(batch_loss.item())
        scheduler.step()  # 更新学习率
        print(f'Epoch {epoch + 1}/{epochs} ====>train_loss {np.mean(epoch_loss):.4f}')
        train_loss.append(np.mean(epoch_loss))

        """测试"""
        net.eval()
        with torch.no_grad():
            epoch_loss_test = []
            epoch_loss_mae_test = []
            epoch_loss_rmse_test = []
            epoch_loss_r2_test = []
            epoch_loss_mape_test = []
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = net(batch_x)
                batch_loss = loss_func(out, batch_y)
                epoch_loss_test.append(batch_loss.item())
                epoch_loss_mae_test.append(MAE(out, batch_y).item())
                epoch_loss_rmse_test.append(RMSE(out, batch_y).item())
                epoch_loss_r2_test.append(R2(out, batch_y).item())
                epoch_loss_mape_test.append(MAPE(out, batch_y).item())
            print(
                f'Epoch {epoch + 1}/{epochs} ====>test_loss={np.mean(epoch_loss_test):.4f} test_MAE={np.mean(epoch_loss_mae_test):.4f} test_RMSE={np.mean(epoch_loss_rmse_test):.4f} R2={np.mean(epoch_loss_r2_test):.4f} MAPE={np.mean(epoch_loss_mape_test):.4f} ')
            test_loss.append(np.mean(epoch_loss_test))
    torch.save(net, "FJJ.pth")
    """可视化"""

    from itertools import chain
    from scipy.stats import gaussian_kde

    l1 = list(out)
    onid1 = list(chain.from_iterable(l1))
    l2 = list(batch_y)
    onid2 = list(chain.from_iterable(l2))
    sample = list(range(0, len(onid2), 15))

    onid2_cpu = [item.cpu() for item in onid2]
    x = np.array(onid2_cpu)[sample]
    onid1_cpu = [item.cpu() for item in onid1]
    y = np.array(onid1_cpu)[sample]

    xy = np.vstack([x, y])

    from scipy import stats

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

    z = stats.gaussian_kde(xy)(xy)

    density_threshold = 0.009
    mask = z > density_threshold
    x, y, z = x[mask], y[mask], z[mask]

    fig, ax = plt.subplots(figsize=(10, 12))  # 留出更多空间给颜色棒
    ax.set_aspect('equal')

    sc = ax.scatter(x, y, c=z, s=20, cmap='plasma', edgecolor='none')

    shared_bins = np.histogram_bin_edges(x, bins=128)

    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 1.0, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.0, pad=0.1, sharey=ax)

    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax_histx.hist(x, bins=shared_bins, density=True, alpha=0.75, color='gray')
    ax_histy.hist(y, bins=shared_bins, orientation='horizontal', density=True, alpha=0.75, color='gray')

    # 设置直方图的刻度字体大小
    ax_histx.tick_params(axis='x', labelsize=22)  # 调整上方直方图的 x 轴刻度字体大小
    ax_histx.tick_params(axis='y', labelsize=22)  # 调整上方直方图的 y 轴刻度字体大小

    ax_histy.tick_params(axis='x', labelsize=22)  # 调整右侧直方图的 x 轴刻度字体大小
    ax_histy.tick_params(axis='y', labelsize=22)  # 调整右侧直方图的 y 轴刻度字体大小

    ax.set_xlabel('S3-Zsd (m)', fontsize=28)
    ax.set_ylabel('HS2-Zsd (m)', fontsize=28)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)

    # 获取当前的刻度标签
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    # 获取当前的刻度标签文本
    x_tick_labels = [f'{tick:.0f}' for tick in x_ticks]
    y_tick_labels = [f'{tick:.0f}' for tick in y_ticks]

    # 移除最后一个刻度标签
    x_tick_labels[-1] = ''
    y_tick_labels[-1] = ''

    # 应用新的刻度标签
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=25)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=25)

    ax.plot([np.min([x, y]), np.max([x, y])], [np.min([x, y]), np.max([x, y])], 'k--', linewidth=3)

    # 添加颜色栏在底部，调整位置和尺寸
    axins = inset_axes(ax, width="100%", height="5%", loc='lower center',
                       bbox_to_anchor=(0, -0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=axins, orientation='horizontal')
    cbar.set_label('Density', rotation=0, labelpad=10, fontsize=22)
    cbar.set_ticks([np.min(z), np.mean(z), np.max(z)])
    cbar.set_ticklabels([f'{np.min(z):.2f}', f'{np.mean(z):.2f}', f'{np.max(z):.2f}'])
    cbar.ax.tick_params(labelsize=20)

    # 设置文本字符串，包括一个百分比
    N = 81552
    r2 = 0.93
    mapd = 12.27  # 作为百分比表示
    mae = 0.52
    # 格式化文本字符串
    text_str = f'N: {N}\n$R^2$: {r2:.2f}\nMAPD: {mapd:.1f}%\nMAE: {mae:.2f} m'

    # 将文本添加到图表
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=28,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))

    # 可选添加图标题
    ax.set_title('Scatter Plot with Marginal Histograms', fontsize=28, pad=20)

    # 保存图像
    path = "C:\\Users\\qiao hanyang\\Desktop\\Paper\\paperFIG\\"
    plt.savefig(path + 'Train-HS2.png', dpi=900, bbox_inches='tight')

    # 显示图像
    plt.show()