import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import os

def compute_4d_normals_pca(points_4d, k_neighbors=15):
    """
    使用PCA方法计算4D点云(xyz-t)的法向量
    
    Args:
        points_4d: 4D点云数据 shape=(n,4)，包含x,y,z,t四个维度
        k_neighbors: 近邻点数量
        
    Returns:
        normals: 4D法向量数组 shape=(n,4)
    """
    # 构建KD树用于快速查找近邻
    tree = KDTree(points_4d)
    
    # 存储法向量
    normals = np.zeros_like(points_4d)
    
    # 为每个点计算法向量
    for i in range(len(points_4d)):
        # 查找K近邻点
        distances, indices = tree.query(points_4d[i:i+1], k=k_neighbors)
        
        # 获取近邻点集
        neighborhood = points_4d[indices[0]]
        
        # 计算中心点
        centroid = np.mean(neighborhood, axis=0)
        
        # 计算协方差矩阵
        centered_points = neighborhood - centroid
        covariance = np.dot(centered_points.T, centered_points) / (len(neighborhood) - 1)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 最小特征值对应的特征向量即为法向量
        normals[i] = eigenvectors[:, 0]
        
    return normals

def visualize_normals(points_4d, normals_4d, save_path, time_slice_idx=None):
    """
    可视化4D法向量在各个投影平面上的投影并保存
    
    Args:
        points_4d: 4D点云数据 shape=(n,4)
        normals_4d: 4D法向量数组 shape=(n,4)
        save_path: 保存路径
        time_slice_idx: 当前时间切片索引（用于标题显示）
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 准备投影数据
    projections = [
        ('XY', 0, 1, 'X', 'Y'),
        ('XZ', 0, 2, 'X', 'Z'),
        ('YZ', 1, 2, 'Y', 'Z'),
        ('XT', 0, 3, 'X', 'Time'),
        ('YT', 1, 3, 'Y', 'Time'),
        ('ZT', 2, 3, 'Z', 'Time')
    ]
    
    for i, (title, dim1, dim2, xlabel, ylabel) in enumerate(projections, 1):
        ax = fig.add_subplot(2, 3, i)
        
        # 绘制点
        ax.scatter(points_4d[:, dim1], points_4d[:, dim2], c='gray', s=1, alpha=0.3)
        
        # 绘制法向量（每隔n个点画一个，避免太密集）
        stride = max(1, len(points_4d) // 200)  # 控制显示的法向量数量
        
        # 计算法向量投影的比例因子（调整长度）
        scale = 0.05  # 可以调整这个值来改变法向量长度
        
        # 绘制法向量
        for j in range(0, len(points_4d), stride):
            # 获取起点坐标
            x_start = points_4d[j, dim1]
            y_start = points_4d[j, dim2]
            
            # 计算终点坐标（法向量投影）
            x_end = x_start + normals_4d[j, dim1] * scale
            y_end = y_start + normals_4d[j, dim2] * scale
            
            # 绘制法向量箭头
            ax.arrow(x_start, y_start, 
                    x_end - x_start, y_end - y_start,
                    head_width=0.01, head_length=0.02, 
                    fc='red', ec='red', alpha=0.6)
        
        ax.set_title(f'{title} Projection')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 保持投影的纵横比相等
        ax.set_aspect('equal')
    
    if time_slice_idx is not None:
        plt.suptitle(f'Normal Vectors Visualization (Time Slice {time_slice_idx})')
    else:
        plt.suptitle('Normal Vectors Visualization')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 修改保存路径
    if time_slice_idx is not None:
        plt.savefig(os.path.join(save_path, f'normals_slice_{time_slice_idx}.png'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_path, 'normals_all.png'), 
                   dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 创建结果保存目录
    result_dir = 'script_find_t_points/result'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'visualizations'), exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('data/7303_xyzet.csv')
    
    # 数据预处理和归一化
    print("数据预处理...")
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    t = df['t'].values
    
    # 归一化
    scaler = MinMaxScaler()
    x_normalized = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    z_normalized = scaler.fit_transform(z.reshape(-1, 1)).flatten()
    t_normalized = scaler.fit_transform(t.reshape(-1, 1)).flatten()
    
    # 构建4D点云
    points_4d = np.column_stack((x_normalized, y_normalized, z_normalized, t_normalized))
    
    # 设置时间切片参数
    n_slices = 20
    time_step = 1.0 / n_slices
    
    # 存储所有切片的法向量
    all_normals = []
    all_slice_indices = []
    
    # 按时间切片处理
    print("计算法向量...")
    for i in range(n_slices):
        start_time = i * time_step
        end_time = (i + 1) * time_step
        
        print(f"\n处理时间切片 {i+1}/{n_slices} ({start_time:.2f} - {end_time:.2f})")
        
        # 获取当前时间切片的点
        slice_indices = np.where((points_4d[:, 3] >= start_time) & 
                               (points_4d[:, 3] < end_time))[0]
        
        if len(slice_indices) >= 15:  # 确保有足够的点计算法向量
            # 提取切片内的点
            slice_points = points_4d[slice_indices]
            
            # 计算法向量
            slice_normals = compute_4d_normals_pca(slice_points, k_neighbors=15)
            
            # 保存结果
            all_normals.append(slice_normals)
            all_slice_indices.append(slice_indices)
            
            # 生成可视化
            visualize_normals(slice_points, slice_normals, 
                            os.path.join(result_dir, 'visualizations'), 
                            i+1)
            
            print(f"时间切片 {i+1} 包含 {len(slice_indices)} 个点")
    
    # 保存计算结果
    results = {
        'points_4d': points_4d,
        'time_slices': {
            'indices': all_slice_indices,
            'normals': all_normals,
            'n_slices': n_slices,
            'time_step': time_step
        }
    }
    
    # 保存为NPY文件
    np.save(os.path.join(result_dir, 'normals_results.npy'), results)
    print(f"\n结果已保存至 {os.path.join(result_dir, 'normals_results.npy')}")

if __name__ == "__main__":
    main()
