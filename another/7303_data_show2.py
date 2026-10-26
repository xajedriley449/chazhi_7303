import cv2
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

config = {
        "font.family": 'serif',         # 使用衬线字体族
        "font.serif": ['SimSun','Times New Roman'],        # 中文优先使用宋体
        "mathtext.fontset": 'stix',      # 数学公式用STIX字体（兼容Times New Roman）
        "axes.unicode_minus": False,      # 解决负号显示问题
        "font.size":14
        }
rcParams.update(config)

# 生成模拟数据（替换为真实数据）
# x,y,z,e,t
data_path='data/7303_xyzet.csv'
df=pd.read_csv(data_path)
xyz=df[['x','y','z']].values
et=df[['e','t']].values

target_len=500
step = len(xyz) / target_len
indices = np.round(np.arange(0, len(xyz), step)).astype(int)
et_selected=et[indices]#[6000:6430]
xyz_selected=xyz[indices]#[6000:6430]

#points=MinMaxScaler(feature_range=(0,100)).fit_transform(xyz_selected)
points=xyz_selected

num_points = len(points)
x = points[:,0]     # X坐标（单位：米）
y = points[:,1]     # Y坐标（分）
z = points[:,2]     # Z坐标（顶底板50M区域）
dates = MinMaxScaler(feature_range=(0,1)).fit_transform(et_selected[:,1].reshape(-1,1))  # 发震日期归一化
energy = et_selected[:,0]  # 能量（T）

X,Y,Z=xyz[:,0],xyz[:,1],xyz[:,2]

# 创建3D画布
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection='3d')

plt.title("7303工作面微震事件时空强分布", y=0.95)
import os
# 读取工作面图片（动态获取绝对路径，避免由于运行目录引起的找不到文件）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
img_path = os.path.join(parent_dir, "data", "img", "GongZuoMian2_cropped_4_point.png")

if not os.path.exists(img_path):
    raise FileNotFoundError(f"【背景图缺失】请检查图片是否存在: {img_path}")

img = cv2.imread(img_path)  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道


# 定义XY平面线段端点（z值固定为0）
zz=Z.min()
segments = [
    [[1338 +20401038, -325 +3917025, zz], [1260 +20401038, -138 +3917025, zz]],  # l1
    [[1260 +20401038, -138 +3917025, zz], [382 +20401038, -615 +3917025, zz]],  # l2
    [[382 +20401038, -615 +3917025, zz], [462 +20401038, -788 +3917025, zz]],  # l3
    [[462 +20401038, -788 +3917025, zz], [910 +20401038, -545 +3917025, zz]],  # l4
    [[910 +20401038, -545 +3917025, zz], [939 +20401038, -600 +3917025, zz]],  # l5
    [[939 +20401038, -600 +3917025, zz], [1338 +20401038, -388 +3917025, zz]]  # l6
]
bs=[20401038,3917025] # base
# 绘制黑色线段
for seg in segments:
    lx = [seg[0][0], seg[1][0]]
    ly = [seg[0][1], seg[1][1]]
    lz = [seg[0][2], seg[1][2]]
    ax.plot(lx, ly, lz, color='black', linewidth=2, linestyle='-')
s=energy/10
# 1. 主散点图（颜色渐变 + 大小区分）
cmap=LinearSegmentedColormap.from_list('custom', ['black', 'red'])
sc = ax.scatter(x, y, z, 
                c=dates, 
                cmap=cmap,
                s=s,  # 缩放能量值控制点的大小
                alpha=0.8, edgecolors='k', 
                     linewidths=0.3, depthshade=True)

# 2. 投影到各平面（红/绿/蓝）
ax.scatter(x, y, zs=Z.min(), zdir='z', c='orange',s=s, alpha=0.2, label='XY投影')    # XY平面
ax.scatter(x, z, zs=Y.max(), zdir='y', c='green',s=s, alpha=0.2, label='XZ投影')  # XZ平面
ax.scatter(y, z, zs=X.min(), zdir='x', c='blue',s=s, alpha=0.2, label='YZ投影')   # YZ平面

ax.view_init(elev=15, azim=-45)

ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # 禁用y轴科学计数法[2,6](@ref)
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # 禁用x轴科学计数法
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
plt.xticks(fontproperties='Times New Roman')
plt.yticks(fontproperties='Times New Roman')
z_labels=ax.get_zticklabels()
[label.set_fontname('Times New Roman') for label in z_labels]
# ax.set_zticks(zticks)

# 坐标轴标签
ax.set_xlabel(r'$\mathrm{X/(m)}$', labelpad=10)
ax.set_ylabel(r'$\mathrm{Y/(m)}$', labelpad=10)
ax.set_zlabel(r'$\mathrm{Z/(m)}$', labelpad=10)

# 网格与边框样式
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False  # 透明背景
ax.grid(True, color='black', linestyle='--', alpha=0.3)

# 限制坐标范围（匹配原图）
ax.set_aspect('equal', adjustable='datalim')  # 等比例且数据驱动范围
ax.set_zlim3d(Z.min(), Z.max())  # 确保Z轴范围包含投影平面
ax.set_ylim3d(Y.min(), Y.max())  # Y轴范围包含XZ投影平面
ax.set_xlim3d(X.min(), X.max())   # X轴范围包含YZ投影平面

# 橙色方位指示线（示例）
#ax.plot([-100, 100], [0, 0], [0, 0], color='orange', linestyle='--', linewidth=2)

# 1. 时间颜色条
# 在左侧添加颜色条坐标轴（位置参数：[左边界, 下边界, 宽度, 高度]）
cax = fig.add_axes([0.10, 0.15, 0.02, 0.605])  # 左边界5%画布宽度，高度70%
cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ticks=[0,1/6,2/6,3/6,4/6,5/6,1.0],
                    ax=ax, pad=0.1, aspect=30, cax=cax)
cbar.set_ticklabels(['2023-07-05', '2023-08-05', '2023-09-05','2023-10-05', '2023-11-05', '2023-12-05','2024-01-05'], fontproperties='Times New Roman')  # 直接设置标签[4](@ref)
cbar.set_label('微震发生时间/(年-月-日)\n\n\n', rotation=0, labelpad=5, loc='top',va='bottom')

# 2. 能量图例（右侧三列布局）
legend_ax = fig.add_axes([0.83, 0.15, 0.12, 0.6])  # 手动定位图例区域
legend_ax.axis('off')

# 投影示例（红/绿/蓝球）
legend_ax.text(0.099, 1.5, '\n\n\n微震位置投影', va='center')
legend_ax.scatter([0.1], [1.25], s=1000, c='orange')#, edgecolor='k')
legend_ax.text(0.102, 1.25, r'$\mathrm{XY}$方向', va='center')
legend_ax.scatter([0.1], [1], s=1000, c='green')#, edgecolor='k')
legend_ax.text(0.102, 1, r'$\mathrm{XZ}$方向', va='center')
legend_ax.scatter([0.1], [0.75], s=1000, c='blue')#, edgecolor='k')
legend_ax.text(0.102, 0.75, r'$\mathrm{YZ}$方向', va='center')
legend_ax.text(0.099, 0.5, '\n\n\n'+r'微震能量/($\mathrm{J}$)', va='center')#'\n\n\n微震能量/(J)'

# 能量示例（不同大小球体）
sizes = [1e2, 1e3, 1e4]
for i, size in enumerate(sizes):
    legend_ax.scatter([0.1], [0.25 - i*0.25], s=size/10, c='red', edgecolor='k',linewidths=0.3)
legend_ax.text(0.102, 0.25 - 0*0.25, r'$\mathrm{1×10^2}$', va='center')
legend_ax.text(0.102, 0.25 - 1*0.25, r'$\mathrm{1×10^3}$', va='center')
legend_ax.text(0.102, 0.25 - 2*0.25, r'$\mathrm{1×10^4}$', va='center')
#   legend_ax.text(0.25, 0.35 - i*0.15, f'{int(size)} T', va='center')

# 文字标注（工作面信息）
bbox_props = {
    'boxstyle': 'round,pad=0.5',  # 圆角边框，内边距0.5
    'facecolor': 'yellow',        # 填充色
    'edgecolor': 'gray',          # 边框颜色
    'linewidth': 2,               # 边框宽度
    'alpha': 0.5                  # 透明度
}
plt.text(0.0775, -0.035, r'$\mathrm{7303}$工作面', ha='left', va='bottom',
         bbox=bbox_props)

plt.tight_layout()
plt.savefig('another/微震时空强分布图.png', dpi=300, bbox_inches='tight')
plt.show()