import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

# 设置中文字体和格式，满足论文要求
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'], # 中文使用宋体
    'mathtext.fontset': 'stix', # 数学公式和数字使用类似 Times New Roman 的字体
    'axes.formatter.use_mathtext': True, # 强制坐标轴刻度使用数学字体（解决负号乱码）
    'axes.unicode_minus': False, # 禁用 Unicode 负号，配合 mathtext 使用
    'font.size': 12,               
    'axes.labelsize': 14,          
    'xtick.labelsize': 12,         
    'ytick.labelsize': 12,
    'svg.fonttype': 'path',
})

def apply_times_new_roman_ticks(fig, ax, cbar=None):
    """强制将坐标轴和颜色条的刻度标签字体设置为 Times New Roman"""
    fig.canvas.draw() # 强制渲染一次以生成 tick labels
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in tick_labels:
        label.set_fontname('Times New Roman')
    
    if cbar is not None:
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')

def plot_st_kde_principle(out_dir):
    """绘制图 2.14：ST-KDE 原理示意图 (分拆为独立子图)"""
    
    # ==========================================
    # (a) 单点各向异性高斯核扩散 (X-Y 平面)
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(5, 5))
    grid_size = 100
    x = np.linspace(-50, 50, grid_size)
    y = np.linspace(-50, 50, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 模拟各向异性：X轴带宽大，Y轴带宽小
    sigma_x = 15.0
    sigma_y = 5.0
    Z = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    
    im1 = ax1.imshow(Z, extent=[-50, 50, -50, 50], origin='lower', cmap='magma')
    ax1.contour(X, Y, Z, levels=[0.1, 0.5, 0.9], colors='white', alpha=0.5, linestyles='dashed')
    ax1.plot(0, 0, 'w*', markersize=10) # 震源点
    
    ax1.set_xlabel("X轴 (m)")
    ax1.set_ylabel("Y轴 (m)")
    
    apply_times_new_roman_ticks(fig, ax1)
    
    plt.tight_layout()
    save_path_a = os.path.join(out_dir, "fig_2_14_a_single_kernel.png")
    plt.savefig(save_path_a, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # (b) 历史能量衰减机制 (时间轴)
    # ==========================================
    fig, ax2 = plt.subplots(figsize=(6, 5))
    t = np.linspace(-36, 12, 200) # 过去36小时到未来12小时
    sigma_t = 12.0
    
    # 衰减权重曲线
    weight = np.exp(-(np.maximum(-t, 0)**2) / (2 * sigma_t**2))
    weight[t > 0] = 0 # 未来事件不产生影响
    
    ax2.plot(t, weight, 'b-', linewidth=2, label='时间衰减权重 $w_j$')
    ax2.fill_between(t, 0, weight, where=(t <= 0), color='blue', alpha=0.2)
    
    ax2.axvline(0, color='r', linestyle='--', label='当前窗口起点 ($t=0$)')
    ax2.axvline(-sigma_t, color='gray', linestyle=':', label='$-1\sigma_t$')
    ax2.axvline(-3*sigma_t, color='gray', linestyle='-.', label='$-3\sigma_t$ (截断点)')
    
    ax2.set_xlabel("相对时间 $\Delta t$ (h)")
    ax2.set_ylabel("能量保留权重")
    ax2.set_xlim(-40, 10)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    apply_times_new_roman_ticks(fig, ax2)
    
    plt.tight_layout()
    save_path_b = os.path.join(out_dir, "fig_2_14_b_time_decay.png")
    plt.savefig(save_path_b, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # (c) 多点时空叠加效果 (T-X 平面)
    # ==========================================
    fig, ax3 = plt.subplots(figsize=(6, 5))
    t_grid = np.linspace(-24, 24, grid_size)
    x_grid = np.linspace(-50, 50, grid_size)
    T, X_grid = np.meshgrid(t_grid, x_grid)
    
    # 模拟三个微震事件
    events = [
        {'t': -12, 'x': -20, 'e': 1.0},
        {'t': -5, 'x': 10, 'e': 0.8},
        {'t': 10, 'x': -5, 'e': 1.2}
    ]
    
    Z_sum = np.zeros_like(T)
    for ev in events:
        # T轴只向未来扩散（因果性）
        dt = T - ev['t']
        mask = dt >= 0
        Z_ev = np.zeros_like(T)
        Z_ev[mask] = ev['e'] * np.exp(-(dt[mask]**2 / (2 * sigma_t**2) + (X_grid[mask] - ev['x'])**2 / (2 * sigma_x**2)))
        Z_sum += Z_ev
        ax3.plot(ev['t'], ev['x'], 'w*', markersize=10, markeredgecolor='k')
        
    im3 = ax3.imshow(Z_sum, extent=[-24, 24, -50, 50], origin='lower', cmap='magma', aspect='auto')
    ax3.axvline(0, color='r', linestyle='--', linewidth=2)
    ax3.text(2, 40, '当前窗口', color='white', fontsize=12)
    ax3.text(-22, 40, '历史窗口', color='white', fontsize=12)
    
    ax3.set_xlabel("时间 T (h)")
    ax3.set_ylabel("空间 X (m)")
    
    apply_times_new_roman_ticks(fig, ax3)
    
    plt.tight_layout()
    save_path_c = os.path.join(out_dir, "fig_2_14_c_st_superposition.png")
    plt.savefig(save_path_c, dpi=300, bbox_inches='tight')
    plt.close()

def plot_interpolation_comparison(out_dir):
    """绘制插值前后对比图 (分拆为独立子图)"""
    try:
        coord_csv = r"data/7303_xyzet_rotated.csv"
        time_csv = r"data/7303_all.csv"
        
        df = pd.read_csv(coord_csv)
        df.columns = df.columns.str.strip().str.lower()
        df2 = pd.read_csv(time_csv, encoding="gb2312")
        df2.columns = df2.columns.str.strip()
        dt = pd.to_datetime(df2["日期"].astype(str).str.strip() + " " + df2["时间"].astype(str).str.strip(), errors="coerce")
        df["_datetime"] = dt
        df = df.dropna(subset=["_datetime"]).copy()
        df["e"] = pd.to_numeric(df["e"], errors="coerce")
        df["e_log"] = np.log10(1.0 + df["e"].clip(lower=0.0))
        
        # 选取数据最密集的一天
        df['date'] = df['_datetime'].dt.date
        top_date = df['date'].value_counts().idxmax()
        win_df = df[df['date'] == top_date].copy()
        
        x_vals = win_df['x'].values
        y_vals = win_df['y'].values
        e_vals = win_df['e_log'].values
        
        x_min, x_max = x_vals.min() - 50, x_vals.max() + 50
        y_min, y_max = y_vals.min() - 50, y_vals.max() + 50
        
    except Exception as e:
        print(f"⚠️ 无法加载真实数据 ({e})，将使用模拟数据生成对比图。")
        np.random.seed(42)
        N = 200
        x_vals = np.random.normal(0, 50, N)
        y_vals = np.random.normal(0, 20, N)
        e_vals = np.random.exponential(2, N)
        x_min, x_max = -150, 150
        y_min, y_max = -80, 80

    # ==========================================
    # (a) 插值前：离散散点图
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(7, 6))
    sizes = (e_vals / e_vals.max()) * 100 + 10
    scatter = ax1.scatter(x_vals, y_vals, s=sizes, c=e_vals, cmap='turbo', alpha=0.7, edgecolors='none')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("X轴 (m)")
    ax1.set_ylabel("Y轴 (m)")
    cbar1 = plt.colorbar(scatter, ax=ax1, label='对数能量 $\lg(E+1)$ (J)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    apply_times_new_roman_ticks(fig, ax1, cbar1)
    
    plt.tight_layout()
    save_path_scatter = os.path.join(out_dir, "fig_interpolation_before_scatter.png")
    plt.savefig(save_path_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # (b) 插值后：连续热力图
    # ==========================================
    fig, ax2 = plt.subplots(figsize=(7, 6))
    grid_size = 256
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # 使用您代码中的真实物理带宽
    sigma_x = 68.18
    sigma_y = 30.11
    
    Z = np.zeros_like(X)
    for i in range(len(x_vals)):
        Z += e_vals[i] * np.exp(-((X - x_vals[i])**2 / (2 * sigma_x**2) + (Y - y_vals[i])**2 / (2 * sigma_y**2)))
        
    # 简单的分位数映射以增强可视化效果
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    Z_flat = Z.flatten().reshape(-1, 1)
    Z_mapped = qt.fit_transform(Z_flat).reshape(grid_size, grid_size)
    
    im2 = ax2.imshow(Z_mapped, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='turbo', aspect='auto')
    ax2.set_xlabel("X轴 (m)")
    ax2.set_ylabel("Y轴 (m)")
    cbar2 = plt.colorbar(im2, ax=ax2, label='归一化能量密度')
    
    apply_times_new_roman_ticks(fig, ax2, cbar2)
    
    plt.tight_layout()
    save_path_heatmap = os.path.join(out_dir, "fig_interpolation_after_heatmap.png")
    plt.savefig(save_path_heatmap, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    out_dir = r"another/242"
    os.makedirs(out_dir, exist_ok=True)
    
    print("开始生成 ST-KDE 示意图 (独立子图)...")
    plot_st_kde_principle(out_dir)
    plot_interpolation_comparison(out_dir)
    print("全部绘图完成！")
