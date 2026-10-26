import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os

# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, # 解决负号显示问题
    'font.size': 12,               # 全局基础字号
    'axes.labelsize': 18,          # 坐标轴标签字号
    'xtick.labelsize': 16,         # X轴刻度字号
    'ytick.labelsize': 16,         # Y轴刻度字号
    'svg.fonttype': 'path',        # 文字转轮廓
})

def compute_density_points(points_4d, bandwidth=0.01, density_threshold=0.7):
    """
    计算三个轴向投影上的高密度点
    
    Args:
        points_4d: shape=(n,4) 归一化后的4D点云数据(x,y,z,t)
        bandwidth: 核密度估计的带宽
        density_threshold: 密度阈值百分位数(0-1)
    
    Returns:
        dense_points: 标记数组，True表示密度高的点
    """
    n_points = len(points_4d)
    x, y, z = points_4d[:, 0], points_4d[:, 1], points_4d[:, 2]
    
    # 计算每个轴向的核密度
    kde_x = gaussian_kde(x, bw_method=bandwidth)
    kde_y = gaussian_kde(y, bw_method=bandwidth)
    kde_z = gaussian_kde(z, bw_method=bandwidth)
    
    # 计算每个点在各轴向上的密度
    density_x = kde_x(x)
    density_y = kde_y(y)
    density_z = kde_z(z)
    
    # 计算密度阈值
    threshold_x = np.percentile(density_x, density_threshold * 100)
    threshold_y = np.percentile(density_y, density_threshold * 100)
    threshold_z = np.percentile(density_z, density_threshold * 100)
    
    # 在任意一个方向上密度较高的点都被认为是目标点
    dense_points = (
        (density_x > threshold_x) |
        (density_y > threshold_y) |
        (density_z > threshold_z)
    )
    
    return dense_points

def main():
    # 创建结果保存目录
    result_dir = 'script_find_t_points/result'
    os.makedirs(result_dir, exist_ok=True)
    
    img_dir = os.path.join(result_dir, '01_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('data/7303_xyzet_rotated.csv')
    
    # 数据归一化
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(df['x'].values.reshape(-1, 1)).flatten()
    y_normalized = scaler.fit_transform(df['y'].values.reshape(-1, 1)).flatten()
    z_normalized = scaler.fit_transform(df['z'].values.reshape(-1, 1)).flatten()
    t_normalized = scaler.fit_transform(df['t'].values.reshape(-1, 1)).flatten()
    
    # 构建4D点云
    points_4d = np.column_stack((x_normalized, y_normalized, z_normalized, t_normalized))
    
    # 计算高密度点
    print("计算高密度点...")
    dense_points = compute_density_points(
        points_4d,
        bandwidth=0.001,      # 0.01可调参数：控制密度估计的平滑程度
        density_threshold=0.8 # 0.8可调参数：控制密度阈值
    )
    
    # 将分类结果添加到原始数据中
    df['is_dense_point'] = dense_points.astype(int)  # 1表示高密度点，0表示其他点
    
    # 保存结果
    output_path = os.path.join(result_dir, 'density_classification_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n分类结果已保存至: {output_path}")
    
    # 创建颜色数组
    colors = np.where(dense_points, 'red', 'gray')
    
    # 绘制六个视图
    # fig = plt.figure(figsize=(18, 12)) # 这行其实没用到，因为下面循环里重新创建了 fig
    
    # 【修改】：使用 mathtext 格式化坐标轴标签
    projections = [
        ('XY', x_normalized, y_normalized, r'$\mathrm{X}$', r'$\mathrm{Y}$'),
        ('XZ', x_normalized, z_normalized, r'$\mathrm{X}$', r'$\mathrm{Z}$'),
        ('YZ', y_normalized, z_normalized, r'$\mathrm{Y}$', r'$\mathrm{Z}$'),
        ('XT', t_normalized, x_normalized, r'$\mathrm{T}$', r'$\mathrm{X}$'),
        ('YT', t_normalized, y_normalized, r'$\mathrm{T}$', r'$\mathrm{Y}$'),
        ('ZT', t_normalized, z_normalized, r'$\mathrm{T}$', r'$\mathrm{Z}$')
    ]
    
    for i, (title, x_data, y_data, xlabel, ylabel) in enumerate(projections, 1):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(x_data, y_data, c=colors, s=1, alpha=0.6)
        #ax.set_title(f'{title} Projection')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        ax.ticklabel_format(useOffset=False, style='plain')
        
        # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
        fig.canvas.draw() # 强制渲染一次以生成 tick labels
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in tick_labels:
            label.set_fontname('Times New Roman')
            
        plt.tight_layout()
        
        # 保存为 PNG 和 SVG
        save_path_png = os.path.join(img_dir, f'{title}_projection.png')
        save_path_svg = os.path.join(img_dir, f'{title}_projection.svg')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path_png}")
    
    # # 添加图例
    # fig.legend(['Low density', 'High density'], 
    #           ['gray', 'red'], 
    #           loc='upper center', 
    #           bbox_to_anchor=(0.5, 0.05),
    #           ncol=2)
    
    # 打印统计信息
    total_points = len(points_4d)
    dense_point_count = np.sum(dense_points)
    print(f"\n总点数: {total_points}")
    print(f"高密度点数: {dense_point_count}")
    print(f"高密度点占比: {dense_point_count/total_points*100:.2f}%")
    
    # plt.show() # 注释掉，防止阻塞

if __name__ == "__main__":
    main()
