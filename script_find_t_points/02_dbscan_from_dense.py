import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

def apply_dbscan_clustering(df, eps=0.01, min_samples=5):
    # 初始化所有点为噪声点(-1)
    df['cluster'] = -1
    
    # 只对第一阶段识别为连线点的数据进行DBSCAN
    points_of_interest = df[df['is_dense_point'] == 1]
    
    # 准备数据
    X = points_of_interest[['x', 'y', 'z', 't']].values
    
    # 归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # 应用DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_normalized)
    
    # 只更新第一阶段连线点的聚类结果
    df.loc[points_of_interest.index, 'cluster'] = clusters
    
    return df

def plot_projections(df):
    # 生成颜色映射
    unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])  # 排除噪声点
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    
    # 创建颜色字典，确保噪声点为灰色
    color_dict = {-1: '#808080'}  # 使用十六进制灰色
    for i, cluster in enumerate(unique_clusters):
        color_dict[cluster] = colors[i]
    
    # 创建颜色列表
    point_colors = df['cluster'].map(color_dict)
    
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
    # 读取第一阶段结果
    input_path = 'script_find_t_points/result/density_classification_results.csv'
    output_dir = 'script_find_t_points/result'
    df = pd.read_csv(input_path)
    
    # 应用DBSCAN聚类
    df_clustered = apply_dbscan_clustering(df, eps=0.01, min_samples=5)
    
    # 保存结果
    output_path = os.path.join(output_dir, 'dbscan_clustering_results.csv')
    df_clustered.to_csv(output_path, index=False)
    
    # 绘制结果
    fig = plot_projections(df_clustered)
    plot_path = os.path.join(output_dir, 'dbscan_clustering_plot.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {output_path}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()