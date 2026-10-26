import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
from scipy.spatial.distance import cdist

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

def plot_final_projections(df):
    # 生成颜色映射
    unique_clusters = sorted([c for c in df['final_cluster'].unique() if c != -1])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    # 创建颜色字典，噪声点为灰色
    color_dict = {-1: '#808080'}  # 灰色
    for i, cluster in enumerate(unique_clusters):
        color_dict[cluster] = colors[i]
    
    # 创建颜色列表
    point_colors = df['final_cluster'].map(color_dict)
    
    # 创建6个子图
    fig = plt.figure(figsize=(18, 12))
    
    # XY平面
    ax1 = fig.add_subplot(231)
    ax1.scatter(df['x'], df['y'], c=point_colors, s=1)
    ax1.set_title('XY Projection')
    
    # XZ平面
    ax2 = fig.add_subplot(232)
    ax2.scatter(df['x'], df['z'], c=point_colors, s=1)
    ax2.set_title('XZ Projection')
    
    # YZ平面
    ax3 = fig.add_subplot(233)
    ax3.scatter(df['y'], df['z'], c=point_colors, s=1)
    ax3.set_title('YZ Projection')
    
    # XT平面
    ax4 = fig.add_subplot(234)
    ax4.scatter(df['t'], df['x'], c=point_colors, s=1)
    ax4.set_title('TX Projection')
    
    # YT平面
    ax5 = fig.add_subplot(235)
    ax5.scatter(df['t'], df['y'], c=point_colors, s=1)
    ax5.set_title('TY Projection')
    
    # ZT平面
    ax6 = fig.add_subplot(236)
    ax6.scatter(df['t'], df['z'], c=point_colors, s=1)
    ax6.set_title('TZ Projection')
    
    plt.tight_layout()
    return fig

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
    
    # 绘制结果
    fig = plot_final_projections(df)
    plot_path = 'script_find_t_points/result/ransac_plot.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {output_path}")
    print(f"Plot saved to {plot_path}")
    
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