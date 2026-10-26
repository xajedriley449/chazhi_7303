# 第四部分：生成三维空间微震事件分布及其三视图投影（原始坐标 + 对数能量）
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

output_dir = 'another/data_preprocessing copy 2'
os.makedirs(output_dir, exist_ok=True)

# 毕业论文统一图表设置（字体、字号）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'SimSun'],
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 12,
    'svg.fonttype': 'path',        # 文字转轮廓，保证中文显示
})

# 读取数据
df = pd.read_csv('data/7303_xyzet_rotated.csv')  # 替换为实际文件路径


# ------------------ 数据准备 ------------------
# 假设已有 DataFrame df 包含字段：x, y, z, e
x_orig = df['x'].values / 1000.0      # 走向坐标，单位 km
y_orig = df['y'].values / 1000.0      # 倾向坐标，单位 km
z_orig = df['z'].values / 1000.0      # 垂向坐标，单位 km
e_orig = df['e'].values

# 按原始能量从小到大排序（小能量先绘制，大能量后绘制覆盖在上面）
sort_idx = np.argsort(e_orig)
x_orig = x_orig[sort_idx]
y_orig = y_orig[sort_idx]
z_orig = z_orig[sort_idx]
e_orig = e_orig[sort_idx]

# 计算对数能量 E_i = lg(e_i + 1)
E_log = np.log10(e_orig + 1.0)

cmap = plt.get_cmap('turbo')  # 选择颜色映射方案（可选 'viridis', 'plasma' 等）

# 点大小缩放因子（可根据效果调整）
size_scale = 1 #50
# E_log = (E_log - E_log.min()) / (E_log.max() - E_log.min())  # 归一化到 [0, 1]
point_sizes = size_scale * E_log   # 加常数避免最小点过小
size_scale = 0.01 #50
# e_orig = (e_orig - e_orig.min()) / (e_orig.max() - e_orig.min())  # 归一化到 [0, 1]
point_sizes_orig = size_scale * e_orig  # 原始能量点大小（可选）

# 计算各维度的实际物理范围（用于设置等比例）
x_range = x_orig.max() - x_orig.min()
y_range = y_orig.max() - y_orig.min()
z_range = z_orig.max() - z_orig.min()

# 计算一个统一的边距比例（让图形不要顶到边框）
margin = 0.05
x_min, x_max = x_orig.min() - margin * x_range, x_orig.max() + margin * x_range
y_min, y_max = y_orig.min() - margin * y_range, y_orig.max() + margin * y_range
z_min, z_max = z_orig.min() - margin * z_range, z_orig.max() + margin * z_range

# ------------------ 绘图 ------------------
# ------------------ 透明度设置 ------------------
ALPHA = 0.8  # 调整此值控制透明度（0 完全透明，1 完全不透明）

# ========== 版本一：对数能量版 ==========
fig4 = plt.figure(figsize=(16, 12))

# 1. 三维主视图
ax4_main = fig4.add_subplot(2, 2, 1, projection='3d')
scatter4_main = ax4_main.scatter(
    x_orig, y_orig, z_orig,
    s=point_sizes,           # 大小已根据对数能量缩放
    c=E_log,                 # 颜色为对数能量值
    cmap=cmap,              # 颜色映射方案（可选 'jet', 'viridis', 'plasma' 等）
    alpha=ALPHA              # 统一透明度
)

# 适当缩小刻度字体，减少拥挤
#ax4_main.tick_params(axis='both', labelsize=10)

# 设置三维等比例：使X、Y、Z轴的单位长度在屏幕上一致
ax4_main.set_box_aspect([x_range, y_range, z_range])
ax4_main.set_xlim(x_min, x_max)
ax4_main.set_ylim(y_min, y_max)
ax4_main.set_zlim(z_min, z_max)

ax4_main.set_xlabel('走向坐标 X (km)', labelpad=20)
ax4_main.set_ylabel('倾向坐标 Y (km)', labelpad=8)
ax4_main.set_zlabel('垂向坐标 Z (km)')
ax4_main.set_title('微震事件三维空间分布', y=0.85)
cbar = plt.colorbar(scatter4_main, ax=ax4_main, pad=0.15, fraction=0.03)
cbar.set_label('对数能量 lg(e+1) (J)')

# ---------- 关键新增行 ----------
ax4_main.ticklabel_format(useOffset=False, style='plain')   # 三维图也可用，针对X和Y轴
ax4_main.zaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False)) # Z轴需要单独处理
# 或者更简洁地强制三维所有轴：
ax4_main.ticklabel_format(useOffset=False, style='plain', axis='both')
# --------------------------------

# 2. X-Y 投影（俯视图）
ax4_xy = fig4.add_subplot(2, 2, 2)
scatter4_xy = ax4_xy.scatter(x_orig, y_orig, s=point_sizes,
                             c=E_log, cmap=cmap, alpha=ALPHA)
ax4_xy.set_xlabel('走向坐标 X (km)')
ax4_xy.set_ylabel('倾向坐标 Y (km)')
ax4_xy.set_title('X-Y 平面投影 (俯视图)')
ax4_xy.grid(True, linestyle='--', alpha=0.5)
# 设置二维等比例
ax4_xy.set_aspect('equal')
ax4_xy.set_xlim(x_min, x_max)
ax4_xy.set_ylim(y_min, y_max)
ax4_xy.ticklabel_format(useOffset=False, style='plain')  # 新增

# 3. X-Z 投影（走向剖面）
ax4_xz = fig4.add_subplot(2, 2, 3)
scatter4_xz = ax4_xz.scatter(x_orig, z_orig, s=point_sizes,
                             c=E_log, cmap=cmap, alpha=ALPHA)
ax4_xz.set_xlabel('走向坐标 X (km)')
ax4_xz.set_ylabel('垂向坐标 Z (km)')
ax4_xz.set_title('X-Z 平面投影 (走向剖面)')
ax4_xz.grid(True, linestyle='--', alpha=0.5)
ax4_xz.set_aspect('equal')
ax4_xz.set_xlim(x_min, x_max)
ax4_xz.set_ylim(z_min, z_max)
ax4_xz.ticklabel_format(useOffset=False, style='plain')  # 新增

# 4. Y-Z 投影（倾向剖面）
ax4_yz = fig4.add_subplot(2, 2, 4)
scatter4_yz = ax4_yz.scatter(y_orig, z_orig, s=point_sizes,
                             c=E_log, cmap=cmap, alpha=ALPHA)
ax4_yz.set_xlabel('倾向坐标 Y (km)')
ax4_yz.set_ylabel('垂向坐标 Z (km)')
ax4_yz.set_title('Y-Z 平面投影 (倾向剖面)')
ax4_yz.grid(True, linestyle='--', alpha=0.5)
ax4_yz.set_aspect('equal')
ax4_yz.set_xlim(y_min, y_max)
ax4_yz.set_ylim(z_min, z_max)
ax4_yz.ticklabel_format(useOffset=False, style='plain')  # 新增

plt.tight_layout()

# 保存为 SVG 矢量图
save_svg_path = os.path.join(output_dir, 'spatial_3d_projections.svg')
plt.savefig(save_svg_path, format='svg', bbox_inches='tight')
print(f"三维投影图已保存至: {save_svg_path}")

plt.show()





# ========== 补充版本：使用原始能量值 e 控制颜色和大小 ==========
# 重新读取原始数据（避免受版本一排序影响）
x_orig_v2 = df['x'].values / 1000.0
y_orig_v2 = df['y'].values / 1000.0
z_orig_v2 = df['z'].values / 1000.0
e_raw = df['e'].values

# 按原始能量从小到大排序（小能量先绘制，大能量后绘制覆盖在上面）
sort_idx_v2 = np.argsort(e_raw)
x_sorted_v2 = x_orig_v2[sort_idx_v2]
y_sorted_v2 = y_orig_v2[sort_idx_v2]
z_sorted_v2 = z_orig_v2[sort_idx_v2]
e_sorted_v2 = e_raw[sort_idx_v2]

# 采用对数归一化，使颜色和大小分布更均衡
e_log_norm = np.log10(e_sorted_v2 + 1.0)
e_norm = (e_log_norm - e_log_norm.min()) / (e_log_norm.max() - e_log_norm.min())

# 点大小缩放（与上一版本保持一致）
size_scale = 1 #50
point_sizes_raw = point_sizes_orig #size_scale * (e_norm + 0.1)

# 重新计算坐标范围（基于排序后数据，确保与版本一一致）
x_range_v2 = x_sorted_v2.max() - x_sorted_v2.min()
y_range_v2 = y_sorted_v2.max() - y_sorted_v2.min()
z_range_v2 = z_sorted_v2.max() - z_sorted_v2.min()

margin = 0.05
x_min_v2, x_max_v2 = x_sorted_v2.min() - margin * x_range_v2, x_sorted_v2.max() + margin * x_range_v2
y_min_v2, y_max_v2 = y_sorted_v2.min() - margin * y_range_v2, y_sorted_v2.max() + margin * y_range_v2
z_min_v2, z_max_v2 = z_sorted_v2.min() - margin * z_range_v2, z_sorted_v2.max() + margin * z_range_v2

# ------------------ 绘图（原始能量版）------------------
fig5 = plt.figure(figsize=(16, 12))

# 1. 三维主视图
ax5_main = fig5.add_subplot(2, 2, 1, projection='3d')
scatter5_main = ax5_main.scatter(
    x_sorted_v2, y_sorted_v2, z_sorted_v2,
    s=point_sizes_raw,
    c=e_sorted_v2,
    cmap=cmap,
    alpha=ALPHA
)
ax5_main.set_xlabel('走向坐标 X (km)', labelpad=20)
ax5_main.set_ylabel('倾向坐标 Y (km)', labelpad=8)
ax5_main.set_zlabel('垂向坐标 Z (km)')
ax5_main.set_title('微震事件三维空间分布', y=0.85)
ax5_main.set_box_aspect([x_range_v2, y_range_v2, z_range_v2])
ax5_main.set_xlim(x_min_v2, x_max_v2)
ax5_main.set_ylim(y_min_v2, y_max_v2)
ax5_main.set_zlim(z_min_v2, z_max_v2)
cbar = plt.colorbar(scatter5_main, ax=ax5_main, pad=0.15, fraction=0.03)
cbar.set_label('原始能量 e (J)')

# ---------- 关键新增行 ----------
ax5_main.ticklabel_format(useOffset=False, style='plain')   # 三维图也可用，针对X和Y轴
ax5_main.zaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False)) # Z轴需要单独处理
# 或者更简洁地强制三维所有轴：
ax5_main.ticklabel_format(useOffset=False, style='plain', axis='both')
# --------------------------------

# 2. X-Y 投影（使用排序后数据）
ax5_xy = fig5.add_subplot(2, 2, 2)
scatter5_xy = ax5_xy.scatter(
    x_sorted_v2, y_sorted_v2,
    s=point_sizes_raw,
    c=e_sorted_v2,
    cmap=cmap,
    alpha=ALPHA
)
ax5_xy.set_xlabel('走向坐标 X (km)')
ax5_xy.set_ylabel('倾向坐标 Y (km)')
ax5_xy.set_title('X-Y 平面投影 (俯视图)')
ax5_xy.grid(True, linestyle='--', alpha=0.5)
ax5_xy.set_aspect('equal')
ax5_xy.set_xlim(x_min_v2, x_max_v2)
ax5_xy.set_ylim(y_min_v2, y_max_v2)
ax5_xy.ticklabel_format(useOffset=False, style='plain')  # 新增

# 3. X-Z 投影（使用排序后数据）
ax5_xz = fig5.add_subplot(2, 2, 3)
scatter5_xz = ax5_xz.scatter(
    x_sorted_v2, z_sorted_v2,
    s=point_sizes_raw,
    c=e_sorted_v2,
    cmap=cmap,
    alpha=ALPHA
)
ax5_xz.set_xlabel('走向坐标 X (km)')
ax5_xz.set_ylabel('垂向坐标 Z (km)')
ax5_xz.set_title('X-Z 平面投影 (走向剖面)')
ax5_xz.grid(True, linestyle='--', alpha=0.5)
ax5_xz.set_aspect('equal')
ax5_xz.set_xlim(x_min_v2, x_max_v2)
ax5_xz.set_ylim(z_min_v2, z_max_v2)
ax5_xz.ticklabel_format(useOffset=False, style='plain')  # 新增


# 4. Y-Z 投影（使用排序后数据）
ax5_yz = fig5.add_subplot(2, 2, 4)
scatter5_yz = ax5_yz.scatter(
    y_sorted_v2, z_sorted_v2,
    s=point_sizes_raw,
    c=e_sorted_v2,
    cmap=cmap,
    alpha=ALPHA
)
ax5_yz.set_xlabel('倾向坐标 Y (m)')
ax5_yz.set_ylabel('垂向坐标 Z (m)')
ax5_yz.set_title('Y-Z 平面投影 (倾向剖面)')
ax5_yz.grid(True, linestyle='--', alpha=0.5)
ax5_yz.set_aspect('equal')
ax5_yz.set_xlim(y_min_v2, y_max_v2)
ax5_yz.set_ylim(z_min_v2, z_max_v2)
ax5_yz.ticklabel_format(useOffset=False, style='plain')  # 新增


plt.tight_layout()

# 保存 SVG
save_svg_path_raw = os.path.join(output_dir, 'spatial_3d_projections_raw_energy.svg')
plt.savefig(save_svg_path_raw, format='svg', bbox_inches='tight')
print(f"原始能量版三维投影图已保存至: {save_svg_path_raw}")

plt.show()

# ========== 版本三：对数能量大小 + 时间颜色版（真实日期） ==========
import pandas as pd
from datetime import datetime
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# 重新读取原始数据文件（包含日期和时间列）
df_full = pd.read_csv('data/7303_all.csv', encoding='gb2312')  # 请根据实际路径调整

# 合并日期和时间列为 datetime64 对象
df_full['datetime'] = pd.to_datetime(df_full['日期'] + ' ' + df_full['时间'], format='%Y/%m/%d %H:%M:%S')

# 提取坐标（单位 km）和能量，并确保排序一致
x_v3 = df_full['X[m]'].values / 1000.0
y_v3 = df_full['Y[m]'].values / 1000.0
z_v3 = df_full['Z[m]'].values / 1000.0
e_v3 = df_full['能量[J]'].values
time_v3 = df_full['datetime'].values

# 按能量从小到大排序
sort_idx_v3 = np.argsort(e_v3)
x_sorted_v3 = x_v3[sort_idx_v3]
y_sorted_v3 = y_v3[sort_idx_v3]
z_sorted_v3 = z_v3[sort_idx_v3]
e_sorted_v3 = e_v3[sort_idx_v3]
time_sorted_v3 = time_v3[sort_idx_v3]

# 计算对数能量用于点大小
E_log_v3 = np.log10(e_sorted_v3 + 1.0)
point_sizes_v3 = 1.0 * E_log_v3

# 时间转换为 Unix 时间戳（秒）并归一化
time_numeric = time_sorted_v3.astype('int64') // 10**9
time_norm = (time_numeric - time_numeric.min()) / (time_numeric.max() - time_numeric.min())

# 选择颜色映射
cmap_time = plt.get_cmap('viridis')
  # 可改为 'plasma', 'cividis', 'coolwarm' 等

# 重新计算坐标范围
x_range_v3 = x_sorted_v3.max() - x_sorted_v3.min()
y_range_v3 = y_sorted_v3.max() - y_sorted_v3.min()
z_range_v3 = z_sorted_v3.max() - z_sorted_v3.min()
margin = 0.05
x_min_v3, x_max_v3 = x_sorted_v3.min() - margin * x_range_v3, x_sorted_v3.max() + margin * x_range_v3
y_min_v3, y_max_v3 = y_sorted_v3.min() - margin * y_range_v3, y_sorted_v3.max() + margin * y_range_v3
z_min_v3, z_max_v3 = z_sorted_v3.min() - margin * z_range_v3, z_sorted_v3.max() + margin * z_range_v3

# ------------------ 绘图 ------------------
fig6 = plt.figure(figsize=(16, 12))

# 1. 三维主视图
ax6_main = fig6.add_subplot(2, 2, 1, projection='3d')
scatter6_main = ax6_main.scatter(
    x_sorted_v3, y_sorted_v3, z_sorted_v3,
    s=point_sizes_v3,
    c=time_norm,
    cmap=cmap_time,
    alpha=ALPHA
)
ax6_main.set_xlabel('走向坐标 X (km)', labelpad=20)
ax6_main.set_ylabel('倾向坐标 Y (km)', labelpad=8)
ax6_main.set_zlabel('垂向坐标 Z (km)')
ax6_main.set_title('微震事件三维空间分布', y=0.85)
ax6_main.set_box_aspect([x_range_v3, y_range_v3, z_range_v3])
ax6_main.set_xlim(x_min_v3, x_max_v3)
ax6_main.set_ylim(y_min_v3, y_max_v3)
ax6_main.set_zlim(z_min_v3, z_max_v3)
ax6_main.ticklabel_format(useOffset=False, style='plain', axis='both')
ax6_main.zaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

# 2. X-Y 投影
ax6_xy = fig6.add_subplot(2, 2, 2)
scatter6_xy = ax6_xy.scatter(x_sorted_v3, y_sorted_v3, s=point_sizes_v3,
                             c=time_norm, cmap=cmap_time, alpha=ALPHA)
ax6_xy.set_xlabel('走向坐标 X (km)')
ax6_xy.set_ylabel('倾向坐标 Y (km)')
ax6_xy.set_title('X-Y 平面投影 (俯视图)')
ax6_xy.grid(True, linestyle='--', alpha=0.5)
ax6_xy.set_aspect('equal')
ax6_xy.set_xlim(x_min_v3, x_max_v3)
ax6_xy.set_ylim(y_min_v3, y_max_v3)
ax6_xy.ticklabel_format(useOffset=False, style='plain')

# 3. X-Z 投影
ax6_xz = fig6.add_subplot(2, 2, 3)
scatter6_xz = ax6_xz.scatter(x_sorted_v3, z_sorted_v3, s=point_sizes_v3,
                             c=time_norm, cmap=cmap_time, alpha=ALPHA)
ax6_xz.set_xlabel('走向坐标 X (km)')
ax6_xz.set_ylabel('垂向坐标 Z (km)')
ax6_xz.set_title('X-Z 平面投影 (走向剖面)')
ax6_xz.grid(True, linestyle='--', alpha=0.5)
ax6_xz.set_aspect('equal')
ax6_xz.set_xlim(x_min_v3, x_max_v3)
ax6_xz.set_ylim(z_min_v3, z_max_v3)
ax6_xz.ticklabel_format(useOffset=False, style='plain')

# 4. Y-Z 投影
ax6_yz = fig6.add_subplot(2, 2, 4)
scatter6_yz = ax6_yz.scatter(y_sorted_v3, z_sorted_v3, s=point_sizes_v3,
                             c=time_norm, cmap=cmap_time, alpha=ALPHA)
ax6_yz.set_xlabel('倾向坐标 Y (km)')
ax6_yz.set_ylabel('垂向坐标 Z (km)')
ax6_yz.set_title('Y-Z 平面投影 (倾向剖面)')
ax6_yz.grid(True, linestyle='--', alpha=0.5)
ax6_yz.set_aspect('equal')
ax6_yz.set_xlim(y_min_v3, y_max_v3)
ax6_yz.set_ylim(z_min_v3, z_max_v3)
ax6_yz.ticklabel_format(useOffset=False, style='plain')

# ---------- 自定义 Colorbar（真实日期刻度） ----------
# 计算等间距日期刻度
num_ticks = 7
tick_timestamps = np.linspace(time_numeric.min(), time_numeric.max(), num_ticks)
tick_dates = [datetime.fromtimestamp(ts).strftime('%Y/%m/%d') for ts in tick_timestamps]
tick_positions = (tick_timestamps - time_numeric.min()) / (time_numeric.max() - time_numeric.min())

# 创建 ScalarMappable 用于 colorbar
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap=cmap_time)
sm.set_array([])  # 必须调用，否则 colorbar 可能报错

# 使用与版本一相同的参数添加 colorbar
cbar6 = plt.colorbar(sm, ax=ax6_main, pad=0.15, fraction=0.03, ticks=tick_positions)
cbar6.set_ticklabels(tick_dates)
cbar6.set_label('微震发生时间 (年/月/日)', rotation=90, labelpad=15, va='center')
# ----------------------------------------------------

plt.tight_layout()

# 保存 SVG
save_svg_path_time = os.path.join(output_dir, 'spatial_3d_projections_time_color.svg')
plt.savefig(save_svg_path_time, format='svg', bbox_inches='tight')
print(f"时间颜色版三维投影图已保存至: {save_svg_path_time}")

plt.show()