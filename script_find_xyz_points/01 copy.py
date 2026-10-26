"""
点云去噪模块
使用DBSCAN算法标记孤立噪声点（不删除），并输出归一化后的多视图平面对比图
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

# 全局绘图参数（对齐另一个脚本的风格）
# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, # 解决负号显示问题
    'font.size': 12,               
    'axes.labelsize': 18,          
    'xtick.labelsize': 16,         
    'ytick.labelsize': 16,         
    'svg.fonttype': 'path',        # 文字转轮廓
})

def denoise_point_cloud_dbscan(xyz, eps=0.05, min_points=5, verbose=True):
    """
    使用DBSCAN去噪（标记噪声点，不删除）
    """
    if verbose:
        print("\n" + "="*70)
        print("DBSCAN点云去噪")
        print("="*70)
        print(f"参数配置: eps={eps}, min_points={min_points}")
    
    # 归一化坐标以便参数选择
    scaler = StandardScaler()
    xyz_normalized = scaler.fit_transform(xyz)
    
    if verbose:
        print(f"\n执行DBSCAN聚类...")
    
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(xyz_normalized)
    labels = clustering.labels_
    
    mask = labels != -1
    noise_count = np.sum(~mask)
    
    if verbose:
        print(f"\n去噪结果:")
        print(f"  总点数:      {len(xyz):8d}")
        print(f"  噪声点数:    {noise_count:8d} ({100*noise_count/len(xyz):6.2f}%)")
        print(f"  保留点数:    {np.sum(mask):8d} ({100*np.sum(mask)/len(xyz):6.2f}%)")
    
    return mask, labels


def visualize_matplotlib_style(df, result_dir):
    """
    生成归一化红灰配色风格的多视图投影（保留点为红，噪声点为灰）
    """
    img_dir = os.path.join(result_dir, '01_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 坐标归一化
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(df[['x']]).flatten()
    y_norm = scaler.fit_transform(df[['y']]).flatten()
    z_norm = scaler.fit_transform(df[['z']]).flatten()
    
    # 配色：由于红灰风格中红色通常代表我们关注的目标，故保留点为红，噪声为灰
    colors = np.where(df['is_noise'] == 0, 'red', 'gray')
    
    print(f"\n生成红灰风格可视化投影 (红=保留目标点，灰=噪声点)...")
    
    # ================= 1. 三维视图 (X-Y-Z) =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_norm, y_norm, z_norm, c=colors, s=2, alpha=0.6)
    
    # 增加 labelpad 并使用更科学的视距（dist）避免文字挤出画框
    # 【修改】：使用 mathtext 格式化坐标轴标签
    ax.set_xlabel(r'$\mathrm{X}$', labelpad=15)
    ax.set_ylabel(r'$\mathrm{Y}$', labelpad=15)
    ax.set_zlabel(r'$\mathrm{Z}$', labelpad=10)
    
    # 设置 3D 轴数字的间距
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=8)
    
    # 强制盒子的长宽比，防止某一个轴过于扁平导致标签重叠
    ax.set_box_aspect([1, 1, 1])
    
    # 【致胜修改】：调整 azim（方位角）可以控制坐标轴的方向和重叠
    # azim=-45 是最经典的默认视角，能让 Z 轴稳定出现在图像的**右侧**
    ax.view_init(elev=20, azim=-45)
    
    # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
    formatter = plt.ScalarFormatter(useOffset=False)
    formatter.set_useMathText(True)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=12)
    
    # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
    fig.canvas.draw()
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    for label in tick_labels:
        label.set_fontname('Times New Roman')
    
    save_path_png = os.path.join(img_dir, 'XYZ_3D_projection.png')
    save_path_svg = os.path.join(img_dir, 'XYZ_3D_projection.svg')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"已保存: {save_path_png}")

    # ================= 2. 二维视图 (XY, XZ, YZ) =================
    # 【修改】：使用 mathtext 格式化坐标轴标签
    projections = [
        ('XY', x_norm, y_norm, r'$\mathrm{X}$', r'$\mathrm{Y}$'),
        ('XZ', x_norm, z_norm, r'$\mathrm{X}$', r'$\mathrm{Z}$'),
        ('YZ', y_norm, z_norm, r'$\mathrm{Y}$', r'$\mathrm{Z}$')
    ]
    
    for title, px, py, xlabel, ylabel in projections:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(px, py, c=colors, s=2, alpha=0.6)
        
        ax.set_xlabel(f'{xlabel}', labelpad=5)
        ax.set_ylabel(f'{ylabel}', labelpad=5)
        
        # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.tick_params(axis='both', labelsize=14)
        
        # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
        fig.canvas.draw()
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in tick_labels:
            label.set_fontname('Times New Roman')
            
        plt.tight_layout()
        save_path_png = os.path.join(img_dir, f'{title}_projection.png')
        save_path_svg = os.path.join(img_dir, f'{title}_projection.svg')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path_png}")


def denoise_csv_file(input_csv, output_csv, eps=0.05, min_points=5):
    print("="*70)
    print("点云去噪标记处理")
    print("="*70)
    
    if not os.path.exists(input_csv):
        print(f"❌ 错误：文件不存在 {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    xyz_original = df[['x', 'y', 'z']].values.astype(np.float64)
    
    mask, labels = denoise_point_cloud_dbscan(
        xyz_original, eps=eps, min_points=min_points, verbose=True
    )
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df = df.copy()
    result_df['is_noise'] = (~mask).astype(int)  
    result_df['cluster_id'] = labels             
    
    result_df.to_csv(output_csv, index=False)
    print(f"✓ 结果已保存: {output_csv}")
    
    # 掉用替换后的 Matplotlib 输出函数
    visualize_matplotlib_style(result_df, os.path.dirname(output_csv))
    
    return result_df


def main():
    input_file = "data/7303_xyzet_rotated.csv"
    output_file = "script_find_xyz_points/result/01_denoised_points.csv"
    
    denoise_csv_file(
        input_csv=input_file,
        output_csv=output_file,
        eps=0.2,              
        min_points=3          
    )

if __name__ == "__main__":
    main()
