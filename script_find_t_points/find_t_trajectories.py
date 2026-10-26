import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

def find_trajectory_points(points_4d, spatial_threshold=0.05, time_threshold=0.05, 
                         direction_threshold=0.7, min_points=5):
    """
    寻找符合要求的轨迹点
    
    Args:
        points_4d: shape=(n,4) 归一化后的4D点云数据(x,y,z,t)
        spatial_threshold: 空间距离阈值
        time_threshold: 时间距离阈值
        direction_threshold: 方向连续性阈值
        min_points: 最小轨迹点数
    
    Returns:
        trajectory_points: 标记数组，True表示属于轨迹的点
    """
    n_points = len(points_4d)
    trajectory_points = np.zeros(n_points, dtype=bool)
    
    # 构建KD树用于快速查找近邻
    tree = KDTree(points_4d)
    
    # 当前轨迹的点集
    current_trajectory = []
    last_direction = None
    
    # 按时间排序遍历点
    time_sorted_indices = np.argsort(points_4d[:, 3])
    
    for idx in time_sorted_indices:
        current_point = points_4d[idx]
        
        # 查找时空邻域内的点
        neighbors = tree.query_radius([current_point], r=max(spatial_threshold, time_threshold))[0]
        
        # 筛选出时间上在后面的点
        future_neighbors = neighbors[points_4d[neighbors, 3] > current_point[3]]
        
        if len(future_neighbors) == 0:
            # 没有合适的后续点，当前轨迹结束
            if len(current_trajectory) >= min_points:
                trajectory_points[current_trajectory] = True
            current_trajectory = []
            last_direction = None
            continue
        
        # 计算到每个邻居点的时空距离和方向
        spatial_distances = np.sqrt(np.sum((points_4d[future_neighbors, :3] - current_point[:3])**2, axis=1))
        time_distances = points_4d[future_neighbors, 3] - current_point[3]
        
        # 计算方向向量并归一化
        displacement_vectors = points_4d[future_neighbors] - current_point
        norms = np.linalg.norm(displacement_vectors, axis=1, keepdims=True)
        directions = displacement_vectors / np.where(norms > 0, norms, 1)
        
        # 筛选满足条件的邻居点
        valid_mask = (
            (spatial_distances < spatial_threshold) & 
            (time_distances < time_threshold) &
            (np.abs(directions[:, 3]) > direction_threshold)  # 确保方向大致沿着时间轴
        )
        
        valid_neighbors = future_neighbors[valid_mask]
        valid_directions = directions[valid_mask]
        
        if len(valid_neighbors) > 0:
            if not current_trajectory:
                # 新轨迹开始
                current_trajectory = [idx]
                # 选择时间最接近的点作为下一个点
                next_idx = np.argmin(time_distances[valid_mask])
                best_next = valid_neighbors[next_idx]
                current_trajectory.append(best_next)
                last_direction = valid_directions[next_idx]
            else:
                # 如果有上一个方向，选择方向最连续的点
                if last_direction is not None:
                    direction_continuity = np.abs(np.dot(valid_directions, last_direction))
                    best_idx = np.argmax(direction_continuity)
                else:
                    # 如果没有上一个方向，选择时间最接近的点
                    best_idx = np.argmin(time_distances[valid_mask])
                
                best_next = valid_neighbors[best_idx]
                current_trajectory.append(best_next)
                last_direction = valid_directions[best_idx]
        else:
            # 没有合适的后续点，当前轨迹结束
            if len(current_trajectory) >= min_points:
                trajectory_points[current_trajectory] = True
            current_trajectory = []
            last_direction = None
    
    # 处理最后一条轨迹
    if len(current_trajectory) >= min_points:
        trajectory_points[current_trajectory] = True
    
    return trajectory_points

def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('data/7303_xyzet.csv')
    
    # 数据归一化
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(df['x'].values.reshape(-1, 1)).flatten()
    y_normalized = scaler.fit_transform(df['y'].values.reshape(-1, 1)).flatten()
    z_normalized = scaler.fit_transform(df['z'].values.reshape(-1, 1)).flatten()
    t_normalized = scaler.fit_transform(df['t'].values.reshape(-1, 1)).flatten()
    
    # 构建4D点云
    points_4d = np.column_stack((x_normalized, y_normalized, z_normalized, t_normalized))
    
    # 寻找轨迹点
    print("寻找轨迹点...")
    trajectory_points = find_trajectory_points(
        points_4d,
        spatial_threshold=0.05,  # 可调参数
        time_threshold=0.007,     # 可调参数
        direction_threshold=0.7,  # 可调参数
        min_points=5             # 可调参数
    )
    
    # 创建颜色数组
    colors = np.where(trajectory_points, 'red', 'gray')
    
    # 绘制六个视图
    fig = plt.figure(figsize=(18, 12))
    
    projections = [
        ('XY', x_normalized, y_normalized, 'X', 'Y'),
        ('XZ', x_normalized, z_normalized, 'X', 'Z'),
        ('YZ', y_normalized, z_normalized, 'Y', 'Z'),
        ('XT', t_normalized, x_normalized, 'Time', 'X'),
        ('YT', t_normalized, y_normalized, 'Time', 'Y'),
        ('ZT', t_normalized, z_normalized, 'Time', 'Z')
    ]
    
    for i, (title, x_data, y_data, xlabel, ylabel) in enumerate(projections, 1):
        ax = fig.add_subplot(2, 3, i)
        ax.scatter(x_data, y_data, c=colors, s=1)
        ax.set_title(f'{title} Projection')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    # 打印统计信息
    total_points = len(points_4d)
    trajectory_point_count = np.sum(trajectory_points)
    print(f"\n总点数: {total_points}")
    print(f"轨迹点数: {trajectory_point_count}")
    print(f"轨迹点占比: {trajectory_point_count/total_points*100:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()
