"""
点云去噪模块
使用DBSCAN算法标记孤立噪声点（不删除）
"""

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os


def denoise_point_cloud_dbscan(xyz, eps=0.05, min_points=5, verbose=True):
    """
    使用DBSCAN去噪（标记噪声点，不删除）
    
    Args:
        xyz: 点云坐标 (N, 3)
        eps: 邻域半径（归一化坐标下的距离）
        min_points: 最少邻域点数（少于此数的点被视为噪声）
        verbose: 是否打印信息
    
    Returns:
        mask: 点的标记 (N,)，True表示保留点，False表示噪声点
        labels: DBSCAN聚类标签 (N,)，-1表示噪声点
    """
    if verbose:
        print("\n" + "="*70)
        print("DBSCAN点云去噪")
        print("="*70)
        print(f"参数配置:")
        print(f"  eps (邻域半径): {eps}")
        print(f"  min_points (最少邻域点数): {min_points}")
    
    # 归一化坐标以便参数选择
    scaler = StandardScaler()
    xyz_normalized = scaler.fit_transform(xyz)
    
    # DBSCAN聚类
    if verbose:
        print(f"\n执行DBSCAN聚类...")
    
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(xyz_normalized)
    labels = clustering.labels_
    
    # -1表示噪声点，其他是簇标签
    mask = labels != -1
    noise_count = np.sum(~mask)
    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
    
    if verbose:
        print(f"\n去噪结果:")
        print(f"  总点数:      {len(xyz):8d}")
        print(f"  噪声点数:    {noise_count:8d} ({100*noise_count/len(xyz):6.2f}%)")
        print(f"  保留点数:    {np.sum(mask):8d} ({100*np.sum(mask)/len(xyz):6.2f}%)")
        print(f"  聚类数:      {cluster_count:8d}")
    
    return mask, labels


def denoise_csv_file(input_csv, output_csv, eps=0.05, min_points=5, visualize=True):
    """
    对CSV文件中的点云进行去噪标记（保留所有点，只添加标记列）
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径（添加 is_noise 列）
        eps: DBSCAN邻域半径
        min_points: DBSCAN最少邻域点数
        visualize: 是否可视化去噪结果
    """
    print("="*70)
    print("点云去噪标记处理")
    print("="*70)
    
    # 读取数据
    print(f"\n读取数据: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"❌ 错误：文件不存在 {input_csv}")
        return
    
    df = pd.read_csv(input_csv)
    xyz_original = df[['x', 'y', 'z']].values.astype(np.float64)
    
    print(f"总点数: {len(xyz_original)}")
    print(f"\n原始坐标范围:")
    print(f"  X=[{xyz_original[:,0].min():.2f}, {xyz_original[:,0].max():.2f}]")
    print(f"  Y=[{xyz_original[:,1].min():.2f}, {xyz_original[:,1].max():.2f}]")
    print(f"  Z=[{xyz_original[:,2].min():.2f}, {xyz_original[:,2].max():.2f}]")
    
    # 去噪标记
    mask, labels = denoise_point_cloud_dbscan(
        xyz_original, eps=eps, min_points=min_points, verbose=True
    )
    
    # 保存结果（添加标记列，保留所有点）
    print(f"\n保存结果...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    result_df = df.copy()
    result_df['is_noise'] = (~mask).astype(int)  # 1=噪声点，0=保留点
    result_df['cluster_id'] = labels             # DBSCAN聚类标签
    
    result_df.to_csv(output_csv, index=False)
    print(f"✓ 结果已保存: {output_csv}")
    print(f"\n新增列说明:")
    print(f"  is_noise: 0=保留点，1=噪声点")
    print(f"  cluster_id: DBSCAN聚类ID（-1表示噪声点）")
    
    # 可视化
    if visualize:
        print(f"\n生成可视化...")
        visualize_denoise_result(xyz_original, mask)
    
    print("\n" + "="*70)
    print("✓ 去噪标记完成")
    print("="*70)
    
    return result_df


def visualize_denoise_result(xyz_original, mask):
    """
    可视化去噪标记结果
    蓝色: 保留的点
    红色: 标记为噪声的点
    """
    # 保留的点 (蓝色)
    xyz_keep = xyz_original[mask]
    pcd_keep = o3d.geometry.PointCloud()
    pcd_keep.points = o3d.utility.Vector3dVector(xyz_keep)
    pcd_keep.paint_uniform_color([0, 0, 1])  # 蓝色
    
    # 噪声点 (红色)
    xyz_noise = xyz_original[~mask]
    pcd_noise = o3d.geometry.PointCloud()
    pcd_noise.points = o3d.utility.Vector3dVector(xyz_noise)
    pcd_noise.paint_uniform_color([1, 0, 0])  # 红色
    
    # 可视化
    o3d.visualization.draw_geometries(
        [pcd_keep, pcd_noise],
        window_name=f"点云去噪标记结果 (蓝=保留{len(xyz_keep)}, 红=噪声{len(xyz_noise)})",
        width=1200,
        height=800
    )


def main():
    """主函数：处理点云去噪标记"""
    
    # 配置参数
    input_file = "data/7303_xyzet_rotated.csv"
    output_file = "script_find_xyz_points/result/01_denoised_points.csv"
    
    print("\n【DBSCAN参数说明】")
    print("  eps=0.03, min_points=3   : 轻度去噪（保留细节）")
    print("  eps=0.05, min_points=5   : 中度去噪（推荐）")
    print("  eps=0.08, min_points=10  : 强度去噪（平滑效果强）")
    
    # 执行去噪标记
    denoise_csv_file(
        input_csv=input_file,
        output_csv=output_file,
        eps=0.2,              # ⬅️ 修改此参数调整去噪强度
        min_points=3,          # ⬅️ 修改此参数调整去噪强度
        visualize=True
    )


if __name__ == "__main__":
    main()