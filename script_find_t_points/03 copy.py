import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from scipy.spatial.distance import cdist

# 全局绘图参数（严格对齐论文大字号排版）
plt.rcParams.update({
    'font.size': 12,               
    'axes.labelsize': 18,          
    'xtick.labelsize': 16,         
    'ytick.labelsize': 16,         
})


def fit_line_ransac(points, threshold=0.02):
    """对点集使用RANSAC拟合直线"""
    # 使用t作为自变量，xyz作为因变量
    X = points[:, 3].reshape(-1, 1)  # t值
    ransac_x = RANSACRegressor(residual_threshold=threshold)
    ransac_y = RANSACRegressor(residual_threshold=threshold)
    ransac_z = RANSACRegressor(residual_threshold=threshold)
    
    # 拟合三个方向
    ransac_x.fit(X, points[:, 0])
    ransac_y.fit(X, points[:, 1])
    ransac_z.fit(X, points[:, 2])
    
    # 返回直线参数和内点掩码
    params = np.array([
        ransac_x.estimator_.coef_[0], ransac_x.estimator_.intercept_,
        ransac_y.estimator_.coef_[0], ransac_y.estimator_.intercept_,
        ransac_z.estimator_.coef_[0], ransac_z.estimator_.intercept_
    ])
    
    inliers = (ransac_x.inlier_mask_ & 
              ransac_y.inlier_mask_ & 
              ransac_z.inlier_mask_)
              
    return params, inliers

def merge_similar_lines(line_params, threshold=0.1):
    """合并相似的直线"""
    # 计算直线参数之间的距离矩阵
    distances = cdist(line_params, line_params)
    
    # 使用DBSCAN聚类相似的直线
    clustering = DBSCAN(eps=threshold, min_samples=1)
    line_clusters = clustering.fit_predict(line_params)
    
    return line_clusters

def reassign_points(df, line_params, line_clusters):
    """重新分配点到最近的直线"""
    new_clusters = np.full(len(df), -1)
    points = df[['x', 'y', 'z', 't']].values
    
    for i, cluster in enumerate(np.unique(line_clusters)):
        # 获取该类别的直线参数平均值
        cluster_params = line_params[line_clusters == cluster].mean(axis=0)
        
        # 计算点到直线的距离
        t = points[:, 3].reshape(-1, 1)
        predicted_x = cluster_params[0] * t + cluster_params[1]
        predicted_y = cluster_params[2] * t + cluster_params[3]
        predicted_z = cluster_params[4] * t + cluster_params[5]
        
        distances = np.sqrt(
            (points[:, 0] - predicted_x.flatten())**2 +
            (points[:, 1] - predicted_y.flatten())**2 +
            (points[:, 2] - predicted_z.flatten())**2
        )
        
        # 将距离小于阈值的点分配给该直线
        mask = distances < 0.05  # 可调整的阈值
        new_clusters[mask] = i
        
    return new_clusters

def visualize_matplotlib_style(df, result_dir):
    """
    生成归一化多视图投影（不同RANSAC拟合线采用不同颜色，噪声点为灰）
    """
    img_dir = os.path.join(result_dir, '03_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 坐标系空间及时间归一化可视化
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(df[['x']]).flatten()
    y_norm = scaler.fit_transform(df[['y']]).flatten()
    z_norm = scaler.fit_transform(df[['z']]).flatten()
    t_norm = scaler.fit_transform(df[['t']]).flatten() 
    
    unique_clusters = sorted([c for c in df['final_cluster'].unique() if c != -1])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    color_dict = {-1: 'gray'}  # 噪声点为灰色
    for i, cluster in enumerate(unique_clusters):
        color_dict[cluster] = colors[i]
        
    point_colors = df['final_cluster'].map(color_dict).values
    
    print(f"\n生成多彩 RANSAC 拟合线可视化投影...")
    
    # ================= 1. 三维视图 (X-Y-Z) =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_norm, y_norm, z_norm, c=point_colors, s=2, alpha=0.6)
    
    ax.set_xlabel('X', labelpad=25)
    ax.set_ylabel('Y', labelpad=25)
    ax.set_zlabel('Z', labelpad=20)
    
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=8)
    
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-45)  
    ax.tick_params(axis='both', labelsize=12)
    
    save_path = os.path.join(img_dir, 'XYZ_3D_projection.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"已保存: {save_path}")

    # ================= 2. 二维视图 (XY, XZ, YZ, XT, YT, ZT) =================
    projections = [
        ('XY', x_norm, y_norm, 'X', 'Y'),
        ('XZ', x_norm, z_norm, 'X', 'Z'),
        ('YZ', y_norm, z_norm, 'Y', 'Z'),
        ('XT', t_norm, x_norm, 'T', 'X'),
        ('YT', t_norm, y_norm, 'T', 'Y'),
        ('ZT', t_norm, z_norm, 'T', 'Z')
    ]
    
    for title, px, py, xlabel, ylabel in projections:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(px, py, c=point_colors, s=2, alpha=0.6)
        
        ax.set_xlabel(f'{xlabel}', labelpad=10)
        ax.set_ylabel(f'{ylabel}', labelpad=10)
        ax.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        save_path = os.path.join(img_dir, f'{title}_projection.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path}")

def main():
    # 读取DBSCAN的结果
    input_path = 'script_find_t_points/result/dbscan_clustering_results.csv'
    df = pd.read_csv(input_path)
    
    # 为每个簇拟合直线
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    line_params = []
    
    for cluster in unique_clusters:
        cluster_points = df[df['cluster'] == cluster][['x', 'y', 'z', 't']].values
        params, _ = fit_line_ransac(cluster_points)
        line_params.append(params)
    
    line_params = np.array(line_params)
    
    # # 合并相似的直线
    # line_clusters = merge_similar_lines(line_params, threshold=0.01)
    
    # # 重新分配点
    # new_clusters = reassign_points(df, line_params, line_clusters)
    # 将 final_cluster 设置为每条线的编号
    df['final_cluster'] = -1  # 初始化为 -1 表示噪声点
    for i, cluster in enumerate(unique_clusters):
        mask = (df['cluster'] == cluster)  # 使用 cluster 列来标识点所属的簇
        df.loc[mask, 'final_cluster'] = i
    
    # 保存结果
    output_path = 'script_find_t_points/result/ransac_results.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    
    # 【修改】：调用新的标准化可视化函数
    visualize_matplotlib_style(df, 'script_find_t_points/result')
    
    # 输出拟合的线参数
    print("\n" + "="*60)
    print("拟合的线参数:")
    print("="*60)
    
    # line_params 形状为 (n_lines, 6), 每根线有 6 个参数
    # params[1], params[3], params[5] 分别是 x₀, y₀, z₀ 截距
    for i in range(len(line_params)):
        params = line_params[i]
        x0 = params[1]  # x截距
        y0 = params[3]  # y截距
        z0 = params[5]  # z截距
        
        print(f"\n线 {i}:")
        print(f"  x₀ (x截距): {x0:.6f}")
        print(f"  y₀ (y截距): {y0:.6f}")
        print(f"  z₀ (z截距): {z0:.6f}")
    
    print("="*60 + "\n")
    
     # 保存线参数到CSV (仅保存截距和时间轴范围)
    params_output_path = 'script_find_t_points/result/line_parameters.csv'
    params_df = pd.DataFrame([
        {
            'line_id': i,
            'x0': line_params[i][1],  # x截距
            'y0': line_params[i][3],  # y截距
            'z0': line_params[i][5],  # z截距
            't_min': df[df['final_cluster'] == i]['t'].min(),  # 时间轴最小值
            't_max': df[df['final_cluster'] == i]['t'].max()   # 时间轴最大值
        }
        for i in range(len(line_params))
    ])
    params_df.to_csv(params_output_path, index=False)
    print(f"Line parameters saved to {params_output_path}")

if __name__ == "__main__":
    main()