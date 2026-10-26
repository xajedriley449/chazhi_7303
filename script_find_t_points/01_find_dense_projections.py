import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os

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
    fig = plt.figure(figsize=(18, 12))
    
    projections = [
        ('XY', x_normalized, y_normalized, 'X', 'Y'),
        ('XZ', x_normalized, z_normalized, 'X', 'Z'),
        ('YZ', y_normalized, z_normalized, 'Y', 'Z'),
        ('XT', t_normalized, x_normalized, 'T', 'X'),
        ('YT', t_normalized, y_normalized, 'T', 'Y'),
        ('ZT', t_normalized, z_normalized, 'T', 'Z')
    ]
    
    for i, (title, x_data, y_data, xlabel, ylabel) in enumerate(projections, 1):
        ax = fig.add_subplot(2, 3, i)
        ax.scatter(x_data, y_data, c=colors, s=1, alpha=0.6)
        #ax.set_title(f'{title} Projection')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
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
    
    plt.show()

if __name__ == "__main__":
    main()
