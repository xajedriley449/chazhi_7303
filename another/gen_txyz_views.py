import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# ================= 1. 环境与排版初始化 =================
output_dir = 'another/txyz'
os.makedirs(output_dir, exist_ok=True)

# 毕业论文统一图表设置
# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, # 解决负号显示问题
    'font.size': 14,               
    'axes.labelsize': 16,          
    'xtick.labelsize': 16,         
    'ytick.labelsize': 16,
    'svg.fonttype': 'path',        
})

# ================= 2. 数据读取与预处理 =================
print("正在读取微震空间结构与时间数据...")
df = pd.read_csv('data/7303_xyzet_rotated.csv')

# 提取空间坐标并转为 km
x = df['x'].values / 1000.0
y = df['y'].values / 1000.0
z = df['z'].values / 1000.0
e_orig = df['e'].values

# 提取时间并归一化 (使用秒级时间戳)
df_full = pd.read_csv('data/7303_all.csv', encoding='gb2312')
df_full['datetime'] = pd.to_datetime(df_full['日期'] + ' ' + df_full['时间'], format='%Y/%m/%d %H:%M:%S')
time_vals = df_full['datetime'].values.astype('int64') // 10**9

# 【新增修改】：将空间坐标 (x, y, z) 进行 min-max 归一化至 [0, 1] 区间
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())
z_norm = (z - z.min()) / (z.max() - z.min())

# 提取归一化时间作为 T 轴 (为了在空间全为 [0,1] 的情况下突出时间的“细长延伸感”，强行放大了10倍，导致坐标上显示为 0-10)
# 【修改】：现在直接将其保留在 [0, 1] 之间，以匹配空间轴的真实归一化！
t_normalized = (time_vals - time_vals.min()) / (time_vals.max() - time_vals.min()) 
t_display = t_normalized # 取消乘以 10

# 计算对数能量用于映射大小和颜色
E_log = np.log10(e_orig + 1.0)
sort_idx = np.argsort(E_log) # 按能量从小到大排序绘制，防止大点遮挡

# 应用排序与归一化数据
X = x_norm[sort_idx]
Y = y_norm[sort_idx]
Z = z_norm[sort_idx]
T = t_display[sort_idx]
E = E_log[sort_idx]

# 统一算点大小和颜色映射
point_sizes = 1.0 * E  
cmap = plt.get_cmap('turbo')
ALPHA = 0.7

# ================= 3. 绘图辅助函数 =================
def plot_txyz_view(save_name, is_3d, x_data, y_data, z_data, x_label, y_label, z_label=None):
    # 【致胜调整】：将画布明显拉宽，使时间轴水平铺开时看起来像一条具有延伸感的轨迹
    fig = plt.figure(figsize=(10, 5)) 
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x_data, y_data, z_data, s=point_sizes, c=E, cmap=cmap, alpha=ALPHA)
        
        ax.set_xlabel(x_label, labelpad=18)
        ax.set_ylabel(y_label, labelpad=8)
        ax.set_zlabel(z_label, labelpad=5)
        
        # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)
        
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        
        # 【新增：3D 盒子长宽比】：强制拉长 X 轴（此时承载着时间 T），使其展现出时空隧道的细长感
        ax.set_box_aspect((2.5, 1, 1))  
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, fraction=0.03, shrink=0.7)
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(x_data, y_data, s=point_sizes, c=E, cmap=cmap, alpha=ALPHA)
        
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label, labelpad=10)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        ax.ticklabel_format(useOffset=False, style='plain')
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.03, fraction=0.046, shrink=0.8)
    
    # 【修改】：在单位前添加空格
    cbar.set_label(r'对数能量 $\mathrm{lg(e+1)\ (J)}$', labelpad=12, fontsize=16)
    
    # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
    fig.canvas.draw() # 强制渲染一次以生成 tick labels
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    if is_3d:
        tick_labels += ax.get_zticklabels()
    
    for label in tick_labels:
        label.set_fontname('Times New Roman')
        
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    plt.tight_layout()
    
    png_path = os.path.join(output_dir, save_name + '.png')
    svg_path = os.path.join(output_dir, save_name + '.svg')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight')
    print(f"图表已生成: {png_path}")
    plt.close(fig)


# ================= 4. 图 2.5: 三维时空视图 (T-X-Y, T-X-Z, T-Y-Z) =================
# 【核心修改】：统一所有标签为 “归一化”
print("\n--- 正在生成图 2.5: 三维时空演化视图 ---")
# 【修改】：使用 mathtext 格式化坐标轴标签
plot_txyz_view('Fig2.5_a_TXY', True, T, X, Y, r'$\mathrm{T}$', r'$\mathrm{X}$', r'$\mathrm{Y}$')
plot_txyz_view('Fig2.5_b_TXZ', True, T, X, Z, r'$\mathrm{T}$', r'$\mathrm{X}$', r'$\mathrm{Z}$')
plot_txyz_view('Fig2.5_c_TYZ', True, T, Y, Z, r'$\mathrm{T}$', r'$\mathrm{Y}$', r'$\mathrm{Z}$')

# ================= 5. 图 2.6: 二维时间投影视图 (T-X, T-Y, T-Z) =================
print("\n--- 正在生成图 2.6: 二维空间-时间投影 ---")
# 【修改】：使用 mathtext 格式化坐标轴标签
plot_txyz_view('Fig2.6_a_TX', False, T, X, None, r'$\mathrm{T}$', r'$\mathrm{X}$')
plot_txyz_view('Fig2.6_b_TY', False, T, Y, None, r'$\mathrm{T}$', r'$\mathrm{Y}$')
plot_txyz_view('Fig2.6_c_TZ', False, T, Z, None, r'$\mathrm{T}$', r'$\mathrm{Z}$')

print("\n🎉 所有时空横伸展视图(坐标已全归一化)抽取完毕！")
