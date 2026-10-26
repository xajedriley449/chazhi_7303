import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches

# 读取数据
data = pd.read_csv('data/7303_xyzet_rotated.csv')  # 替换为实际文件路径
x = data['x'].values
y = data['y'].values
z = data['z'].values
e = data['e'].values
t = data['t'].values

# 归一化所有坐标值
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_z = MinMaxScaler()
scaler_t = MinMaxScaler()
scaler_e = MinMaxScaler()

# 归一化坐标值
x_normalized = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
z_normalized = scaler_z.fit_transform(z.reshape(-1, 1)).flatten()
t_normalized = scaler_t.fit_transform(t.reshape(-1, 1)).flatten()
e_normalized = scaler_e.fit_transform(e.reshape(-1, 1)).flatten()

# 初始能量缩放因子
initial_energy_scale = 50

#####################################################################################################
# 第一部分：使用原始数据生成图表（归一化坐标）
fig1 = plt.figure(figsize=(18, 12))

# 添加滑动条位置
ax_slider1 = plt.axes([0.25, 0.02, 0.5, 0.03])
slider1 = Slider(ax_slider1, 'Energy Scale', 1, 200, valinit=initial_energy_scale)

# 存储3D散点图对象和数据的字典
scatter3d_dict1 = {}

# 1. x-t 二维曲线图
ax1_1 = plt.subplot(2, 3, 1)
scatter1_1 = ax1_1.scatter(t_normalized, x_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized X Coordinate')
plt.title('X-T Curve (Normalized Coordinates)')
plt.colorbar(scatter1_1, label='Normalized Time')

# 2. y-t 二维曲线图
ax1_2 = plt.subplot(2, 3, 2)
scatter1_2 = ax1_2.scatter(t_normalized, y_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Y Coordinate')
plt.title('Y-T Curve (Normalized Coordinates)')
plt.colorbar(scatter1_2, label='Normalized Time')

# 3. z-t 二维曲线图
ax1_3 = plt.subplot(2, 3, 3)
scatter1_3 = ax1_3.scatter(t_normalized, z_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Z Coordinate')
plt.title('Z-T Curve (Normalized Coordinates)')
plt.colorbar(scatter1_3, label='Normalized Time')

# 4. xz-t 三维轨迹图
ax1_4 = plt.subplot(2, 3, 4, projection='3d')
# 存储数据以便重新创建散点图
scatter3d_dict1['ax1_4'] = {
    'x': x_normalized, 'z': z_normalized, 't': t_normalized,
    'e': e_normalized, 'c': t_normalized, 'ax': ax1_4
}
scatter1_4 = ax1_4.scatter(x_normalized, z_normalized, t_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
ax1_4.set_xlabel('Normalized X Coordinate')
ax1_4.set_ylabel('Normalized Z Coordinate')
ax1_4.set_zlabel('Normalized Time')
ax1_4.set_title('XZ-T 3D Trajectory (Normalized Coordinates)')

# 5. xy-t 三维轨迹图
ax1_5 = plt.subplot(2, 3, 5, projection='3d')
scatter3d_dict1['ax1_5'] = {
    'x': x_normalized, 'y': y_normalized, 't': t_normalized,
    'e': e_normalized, 'c': t_normalized, 'ax': ax1_5
}
scatter1_5 = ax1_5.scatter(x_normalized, y_normalized, t_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
ax1_5.set_xlabel('Normalized X Coordinate')
ax1_5.set_ylabel('Normalized Y Coordinate')
ax1_5.set_zlabel('Normalized Time')
ax1_5.set_title('XY-T 3D Trajectory (Normalized Coordinates)')

# 6. yz-t 三维轨迹图
ax1_6 = plt.subplot(2, 3, 6, projection='3d')
scatter3d_dict1['ax1_6'] = {
    'y': y_normalized, 'z': z_normalized, 't': t_normalized,
    'e': e_normalized, 'c': t_normalized, 'ax': ax1_6
}
scatter1_6 = ax1_6.scatter(y_normalized, z_normalized, t_normalized, 
                          s=e_normalized*initial_energy_scale, 
                          c=t_normalized, cmap='viridis', alpha=0.6)
ax1_6.set_xlabel('Normalized Y Coordinate')
ax1_6.set_ylabel('Normalized Z Coordinate')
ax1_6.set_zlabel('Normalized Time')
ax1_6.set_title('YZ-T 3D Trajectory (Normalized Coordinates)')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 存储2D散点图对象
scatter2d_list1 = [scatter1_1, scatter1_2, scatter1_3]

# 更新函数
def update1(val):
    scale = slider1.val
    
    # 更新2D散点图
    for scatter in scatter2d_list1:
        scatter.set_sizes(e_normalized * scale)
    
    # 更新3D散点图 - 需要重新创建
    for key, data_dict in scatter3d_dict1.items():
        ax = data_dict['ax']
        # 清除当前轴
        ax.cla()
        
        # 重新创建散点图
        if key == 'ax1_4':
            ax.scatter(data_dict['x'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XZ-T 3D Trajectory (Normalized Coordinates)')
        elif key == 'ax1_5':
            ax.scatter(data_dict['x'], data_dict['y'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Y Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XY-T 3D Trajectory (Normalized Coordinates)')
        elif key == 'ax1_6':
            ax.scatter(data_dict['y'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized Y Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('YZ-T 3D Trajectory (Normalized Coordinates)')
    
    fig1.canvas.draw_idle()

slider1.on_changed(update1)

# 添加图例
legend_patch = mpatches.Patch(color='gray', label='Point size represents energy')
fig1.legend(handles=[legend_patch], loc='lower center')

plt.savefig('another/data_preprocessing copy 2/normalized_original_data_plots.png', dpi=300)
#plt.show()

#####################################################################################################
# 第二部分：使用下采样数据生成图表（归一化坐标）

# 下采样：每10个点取一个
downsample_factor = 10
x_down = x_normalized[::downsample_factor]
y_down = y_normalized[::downsample_factor]
z_down = z_normalized[::downsample_factor]
t_down = t_normalized[::downsample_factor]
e_down = e_normalized[::downsample_factor]

# 创建图表
fig2 = plt.figure(figsize=(18, 12))

# 添加滑动条位置
ax_slider2 = plt.axes([0.25, 0.02, 0.5, 0.03])
slider2 = Slider(ax_slider2, 'Energy Scale', 1, 200, valinit=initial_energy_scale)

# 存储3D散点图对象和数据的字典
scatter3d_dict2 = {}

# 1. x-t 二维曲线图（下采样）
ax2_1 = plt.subplot(2, 3, 1)
scatter2_1 = ax2_1.scatter(t_down, x_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized X Coordinate')
plt.title('X-T Curve (Downsampled, Normalized)')
plt.colorbar(scatter2_1, label='Normalized Time')

# 2. y-t 二维曲线图（下采样）
ax2_2 = plt.subplot(2, 3, 2)
scatter2_2 = ax2_2.scatter(t_down, y_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Y Coordinate')
plt.title('Y-T Curve (Downsampled, Normalized)')
plt.colorbar(scatter2_2, label='Normalized Time')

# 3. z-t 二维曲线图（下采样）
ax2_3 = plt.subplot(2, 3, 3)
scatter2_3 = ax2_3.scatter(t_down, z_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Z Coordinate')
plt.title('Z-T Curve (Downsampled, Normalized)')
plt.colorbar(scatter2_3, label='Normalized Time')

# 4. xz-t 三维轨迹图（下采样）
ax2_4 = plt.subplot(2, 3, 4, projection='3d')
scatter3d_dict2['ax2_4'] = {
    'x': x_down, 'z': z_down, 't': t_down,
    'e': e_down, 'c': t_down, 'ax': ax2_4
}
scatter2_4 = ax2_4.scatter(x_down, z_down, t_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
ax2_4.set_xlabel('Normalized X Coordinate')
ax2_4.set_ylabel('Normalized Z Coordinate')
ax2_4.set_zlabel('Normalized Time')
ax2_4.set_title('XZ-T 3D Trajectory (Downsampled, Normalized)')

# 5. xy-t 三维轨迹图（下采样）
ax2_5 = plt.subplot(2, 3, 5, projection='3d')
scatter3d_dict2['ax2_5'] = {
    'x': x_down, 'y': y_down, 't': t_down,
    'e': e_down, 'c': t_down, 'ax': ax2_5
}
scatter2_5 = ax2_5.scatter(x_down, y_down, t_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
ax2_5.set_xlabel('Normalized X Coordinate')
ax2_5.set_ylabel('Normalized Y Coordinate')
ax2_5.set_zlabel('Normalized Time')
ax2_5.set_title('XY-T 3D Trajectory (Downsampled, Normalized)')

# 6. yz-t 三维轨迹图（下采样）
ax2_6 = plt.subplot(2, 3, 6, projection='3d')
scatter3d_dict2['ax2_6'] = {
    'y': y_down, 'z': z_down, 't': t_down,
    'e': e_down, 'c': t_down, 'ax': ax2_6
}
scatter2_6 = ax2_6.scatter(y_down, z_down, t_down, 
                          s=e_down*initial_energy_scale, 
                          c=t_down, cmap='viridis', alpha=0.6)
ax2_6.set_xlabel('Normalized Y Coordinate')
ax2_6.set_ylabel('Normalized Z Coordinate')
ax2_6.set_zlabel('Normalized Time')
ax2_6.set_title('YZ-T 3D Trajectory (Downsampled, Normalized)')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 存储2D散点图对象
scatter2d_list2 = [scatter2_1, scatter2_2, scatter2_3]

# 更新函数
def update2(val):
    scale = slider2.val
    
    # 更新2D散点图
    for scatter in scatter2d_list2:
        scatter.set_sizes(e_down * scale)
    
    # 更新3D散点图 - 需要重新创建
    for key, data_dict in scatter3d_dict2.items():
        ax = data_dict['ax']
        # 清除当前轴
        ax.cla()
        
        # 重新创建散点图
        if key == 'ax2_4':
            ax.scatter(data_dict['x'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XZ-T 3D Trajectory (Downsampled, Normalized)')
        elif key == 'ax2_5':
            ax.scatter(data_dict['x'], data_dict['y'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Y Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XY-T 3D Trajectory (Downsampled, Normalized)')
        elif key == 'ax2_6':
            ax.scatter(data_dict['y'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized Y Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('YZ-T 3D Trajectory (Downsampled, Normalized)')
    
    fig2.canvas.draw_idle()

slider2.on_changed(update2)

# 添加图例
legend_patch = mpatches.Patch(color='gray', label='Point size represents energy')
fig2.legend(handles=[legend_patch], loc='lower center')

plt.savefig('another/data_preprocessing copy 2/normalized_downsampled_data_plots.png', dpi=300)
#plt.show()

#####################################################################################################
# 第三部分：使用插值数据生成图表（归一化坐标）

# 插值处理 - 创建更平滑的曲线但保持相同点数
# 注意：这里我们在原始时间尺度上进行插值，然后归一化
t_interp = np.linspace(t.min(), t.max(), len(t))

# 创建插值函数（使用原始坐标）
f_x = interpolate.interp1d(t, x, kind='cubic')
f_y = interpolate.interp1d(t, y, kind='cubic')
f_z = interpolate.interp1d(t, z, kind='cubic')
f_e = interpolate.interp1d(t, e, kind='cubic')

# 应用插值
x_interp = f_x(t_interp)
y_interp = f_y(t_interp)
z_interp = f_z(t_interp)
e_interp = f_e(t_interp)

# 归一化插值后的坐标值
x_interp_normalized = scaler_x.transform(x_interp.reshape(-1, 1)).flatten()
y_interp_normalized = scaler_y.transform(y_interp.reshape(-1, 1)).flatten()
z_interp_normalized = scaler_z.transform(z_interp.reshape(-1, 1)).flatten()
t_interp_normalized = scaler_t.transform(t_interp.reshape(-1, 1)).flatten()
e_interp_normalized = scaler_e.transform(e_interp.reshape(-1, 1)).flatten()

# 创建图表
fig3 = plt.figure(figsize=(18, 12))

# 添加滑动条位置
ax_slider3 = plt.axes([0.25, 0.02, 0.5, 0.03])
slider3 = Slider(ax_slider3, 'Energy Scale', 1, 200, valinit=initial_energy_scale)

# 存储3D散点图对象和数据的字典
scatter3d_dict3 = {}

# 1. x-t 二维曲线图（插值）
ax3_1 = plt.subplot(2, 3, 1)
scatter3_1 = ax3_1.scatter(t_interp_normalized, x_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized X Coordinate')
plt.title('X-T Curve (Interpolated, Normalized)')
plt.colorbar(scatter3_1, label='Normalized Time')

# 2. y-t 二维曲线图（插值）
ax3_2 = plt.subplot(2, 3, 2)
scatter3_2 = ax3_2.scatter(t_interp_normalized, y_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Y Coordinate')
plt.title('Y-T Curve (Interpolated, Normalized)')
plt.colorbar(scatter3_2, label='Normalized Time')

# 3. z-t 二维曲线图（插值）
ax3_3 = plt.subplot(2, 3, 3)
scatter3_3 = ax3_3.scatter(t_interp_normalized, z_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
plt.xlabel('Normalized Time')
plt.ylabel('Normalized Z Coordinate')
plt.title('Z-T Curve (Interpolated, Normalized)')
plt.colorbar(scatter3_3, label='Normalized Time')

# 4. xz-t 三维轨迹图（插值）
ax3_4 = plt.subplot(2, 3, 4, projection='3d')
scatter3d_dict3['ax3_4'] = {
    'x': x_interp_normalized, 'z': z_interp_normalized, 't': t_interp_normalized,
    'e': e_interp_normalized, 'c': t_interp_normalized, 'ax': ax3_4
}
scatter3_4 = ax3_4.scatter(x_interp_normalized, z_interp_normalized, t_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
ax3_4.set_xlabel('Normalized X Coordinate')
ax3_4.set_ylabel('Normalized Z Coordinate')
ax3_4.set_zlabel('Normalized Time')
ax3_4.set_title('XZ-T 3D Trajectory (Interpolated, Normalized)')

# 5. xy-t 三维轨迹图（插值）
ax3_5 = plt.subplot(2, 3, 5, projection='3d')
scatter3d_dict3['ax3_5'] = {
    'x': x_interp_normalized, 'y': y_interp_normalized, 't': t_interp_normalized,
    'e': e_interp_normalized, 'c': t_interp_normalized, 'ax': ax3_5
}
scatter3_5 = ax3_5.scatter(x_interp_normalized, y_interp_normalized, t_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
ax3_5.set_xlabel('Normalized X Coordinate')
ax3_5.set_ylabel('Normalized Y Coordinate')
ax3_5.set_zlabel('Normalized Time')
ax3_5.set_title('XY-T 3D Trajectory (Interpolated, Normalized)')

# 6. yz-t 三维轨迹图（插值）
ax3_6 = plt.subplot(2, 3, 6, projection='3d')
scatter3d_dict3['ax3_6'] = {
    'y': y_interp_normalized, 'z': z_interp_normalized, 't': t_interp_normalized,
    'e': e_interp_normalized, 'c': t_interp_normalized, 'ax': ax3_6
}
scatter3_6 = ax3_6.scatter(y_interp_normalized, z_interp_normalized, t_interp_normalized, 
                          s=e_interp_normalized*initial_energy_scale, 
                          c=t_interp_normalized, cmap='viridis', alpha=0.6)
ax3_6.set_xlabel('Normalized Y Coordinate')
ax3_6.set_ylabel('Normalized Z Coordinate')
ax3_6.set_zlabel('Normalized Time')
ax3_6.set_title('YZ-T 3D Trajectory (Interpolated, Normalized)')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# 存储2D散点图对象
scatter2d_list3 = [scatter3_1, scatter3_2, scatter3_3]

# 更新函数
def update3(val):
    scale = slider3.val
    
    # 更新2D散点图
    for scatter in scatter2d_list3:
        scatter.set_sizes(e_interp_normalized * scale)
    
    # 更新3D散点图 - 需要重新创建
    for key, data_dict in scatter3d_dict3.items():
        ax = data_dict['ax']
        # 清除当前轴
        ax.cla()
        
        # 重新创建散点图
        if key == 'ax3_4':
            ax.scatter(data_dict['x'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XZ-T 3D Trajectory (Interpolated, Normalized)')
        elif key == 'ax3_5':
            ax.scatter(data_dict['x'], data_dict['y'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized X Coordinate')
            ax.set_ylabel('Normalized Y Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('XY-T 3D Trajectory (Interpolated, Normalized)')
        elif key == 'ax3_6':
            ax.scatter(data_dict['y'], data_dict['z'], data_dict['t'], 
                      s=data_dict['e']*scale, 
                      c=data_dict['c'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('Normalized Y Coordinate')
            ax.set_ylabel('Normalized Z Coordinate')
            ax.set_zlabel('Normalized Time')
            ax.set_title('YZ-T 3D Trajectory (Interpolated, Normalized)')
    
    fig3.canvas.draw_idle()

slider3.on_changed(update3)

# 添加图例
legend_patch = mpatches.Patch(color='gray', label='Point size represents energy')
fig3.legend(handles=[legend_patch], loc='lower center')

plt.savefig('another/data_preprocessing copy 2/normalized_interpolated_data_plots.png', dpi=300)
plt.show()


