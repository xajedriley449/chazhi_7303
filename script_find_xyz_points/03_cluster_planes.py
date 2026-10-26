import numpy as np
import pandas as pd
import os
import sys
import open3d as o3d
import colorsys
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualize_3d_point_cloud_continuous


def parse_plane_params(plane_str):
    """从字符串解析平面参数"""
    if plane_str == "None" or pd.isna(plane_str):
        return None
    try:
        parts = plane_str.split(',')
        return tuple(float(p) for p in parts)
    except:
        return None


def visualize_all_planes_with_points(xyz, plane_inliers, title="平面分割结果"):
    """在一个窗口中可视化所有平面及其对应的点云"""
    print("\n" + "="*70)
    print(f"3D点云可视化：{title}")
    print("="*70)

    geometries = []
    
    xyz = np.asarray(xyz)
    n_points = xyz.shape[0]

    # 获取所有有效平面（点数>=5）
    inlier_counts = np.array([len(indices) for indices in plane_inliers]) if len(plane_inliers) > 0 else np.array([])
    valid_plane_ids = np.where(inlier_counts >= 5)[0] if len(inlier_counts) > 0 else np.array([], dtype=int)

    if len(valid_plane_ids) == 0:
        print("❌ 没有有效的平面")
        return

    # 按点数从多到少排序
    sorted_plane_ids = valid_plane_ids[np.argsort(inlier_counts[valid_plane_ids])[::-1]]

    # 非平面点（灰色）
    all_plane_inliers = np.unique(
        np.concatenate([plane_inliers[i] for i in valid_plane_ids])
    ) if len(valid_plane_ids) > 0 else np.array([], dtype=int)

    non_plane_mask = np.ones(n_points, dtype=bool)
    non_plane_mask[all_plane_inliers] = False
    non_plane_indices = np.where(non_plane_mask)[0]

    if len(non_plane_indices) > 0:
        non_plane_pcd = o3d.geometry.PointCloud()
        non_plane_pcd.points = o3d.utility.Vector3dVector(xyz[non_plane_indices])
        non_plane_pcd.paint_uniform_color([0.65, 0.65, 0.65])
        geometries.append(non_plane_pcd)
        print(f"  非平面点数: {len(non_plane_indices)} / {n_points}")
    else:
        print("  非平面点数: 0（所有点都属于某个平面）")

    # 平面点云（彩色）
    num_planes = len(sorted_plane_ids)
    for rank, pid in enumerate(sorted_plane_ids):
        indices = np.asarray(plane_inliers[pid], dtype=int)
        if indices.size == 0:
            continue

        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(xyz[indices])

        hue = (rank / max(num_planes, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        plane_pcd.paint_uniform_color([r, g, b])

        geometries.append(plane_pcd)

    # 可视化
    o3d.visualization.draw_geometries(geometries, window_name=title)
    
# ... 前面的代码保持不变 ...

def visualize_segmented_planes_only_keep(xyz_all, is_noise, plane_inliers, 
                                        title="平面分割结果（仅保留点）"):
    """✅ 仅显示保留点的平面分割结果（不显示噪声点）"""
    print("\n" + "="*70)
    print(f"可视化（仅保留点）：{title}")
    print("="*70)
    
    # 筛选保留点
    keep_mask = is_noise == 0
    keep_indices = np.where(keep_mask)[0]
    
    # ✅ 创建原始索引到保留点索引的映射
    index_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(keep_indices)}
    
    xyz_keep = np.asarray(xyz_all)[keep_indices]
    n_points_keep = len(xyz_keep)
    
    geometries = []
    
    # 获取所有有效平面
    inlier_counts = np.array([len(indices) for indices in plane_inliers]) if len(plane_inliers) > 0 else np.array([])
    valid_plane_ids = np.where(inlier_counts >= 5)[0] if len(inlier_counts) > 0 else np.array([], dtype=int)
    
    if len(valid_plane_ids) == 0:
        print("❌ 没有有效的平面")
        return
    
    # 按点数从多到少排序
    sorted_plane_ids = valid_plane_ids[np.argsort(inlier_counts[valid_plane_ids])[::-1]]
    
    # ✅ 将原始索引映射到保留点索引
    plane_inliers_mapped = []
    for indices in plane_inliers:
        mapped_indices = np.array([index_mapping[idx] for idx in indices if idx in index_mapping])
        plane_inliers_mapped.append(mapped_indices)
    
    # 非平面点（灰色）
    all_plane_inliers = np.unique(
        np.concatenate([plane_inliers_mapped[i] for i in valid_plane_ids])
    ) if len(valid_plane_ids) > 0 else np.array([], dtype=int)
    
    non_plane_mask = np.ones(n_points_keep, dtype=bool)
    non_plane_mask[all_plane_inliers] = False
    non_plane_indices = np.where(non_plane_mask)[0]
    
    if len(non_plane_indices) > 0:
        non_plane_pcd = o3d.geometry.PointCloud()
        non_plane_pcd.points = o3d.utility.Vector3dVector(xyz_keep[non_plane_indices])
        non_plane_pcd.paint_uniform_color([0.65, 0.65, 0.65])
        geometries.append(non_plane_pcd)
        print(f"  非平面点数: {len(non_plane_indices)} / {n_points_keep}")
    else:
        print("  非平面点数: 0（所有保留点都属于某个平面）")
    
    # 平面点云（彩色）
    num_planes = len(sorted_plane_ids)
    for rank, pid in enumerate(sorted_plane_ids):
        indices = np.asarray(plane_inliers_mapped[pid], dtype=int)
        if indices.size == 0:
            continue
        
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(xyz_keep[indices])
        
        hue = (rank / max(num_planes, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        plane_pcd.paint_uniform_color([r, g, b])
        
        geometries.append(plane_pcd)
        print(f"  平面 {pid}: {len(indices)} 个点")
    
    o3d.visualization.draw_geometries(geometries, window_name=title)

def fit_plane_least_squares(points):
    """使用最小二乘法拟合平面（SVD方法）"""
    if len(points) < 3:
        return None
    
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    U, S, Vt = np.linalg.svd(points_centered)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)
    
    a, b, c = normal
    d = -np.dot(normal, centroid)
    
    return (a, b, c, d)


def point_to_plane_distance(point, plane):
    """计算点到平面的距离（原始坐标系）"""
    if plane is None:
        return np.nan
    
    a, b, c, d = plane
    x, y, z = point
    
    numerator = abs(a*x + b*y + c*z + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    
    if denominator < 1e-10:
        return np.nan
    
    return numerator / denominator


def calculate_distances_to_segmented_planes(df, xyz_original, keep_indices, 
                                           valid_plane_clusters, scaler):
    """计算所有点（包括噪声点）到分割平面的距离"""
    print("\n" + "="*70)
    print("计算点到分割平面的距离（原始坐标系）")
    print("="*70)
    
    xyz_surface = df.loc[keep_indices, ['x', 'y', 'z']].values.astype(np.float64)
    
    distances = np.full(len(df), np.nan, dtype=np.float64)
    plane_params_segmented = [None] * len(df)
    
    # ✅ 为每个聚类平面拟合最终的平面参数（基于原始坐标）
    plane_dict = {}
    
    for plane_id, local_indices in valid_plane_clusters.items():
        local_indices = np.array(local_indices)
        original_indices = keep_indices[local_indices]
        
        # 使用原始坐标拟合平面
        points_original = xyz_surface[local_indices]
        plane = fit_plane_least_squares(points_original)
        plane_dict[plane_id] = plane
        
        if plane is not None:
            plane_str = f"{plane[0]:.6f},{plane[1]:.6f},{plane[2]:.6f},{plane[3]:.6f}"
            
            # ✅ 计算这个平面内所有点到平面的距离
            for local_idx, orig_idx in zip(local_indices, original_indices):
                distances[orig_idx] = point_to_plane_distance(
                    xyz_surface[local_idx], plane
                )
                plane_params_segmented[orig_idx] = plane_str
    
    # ✅ 新增：为噪声点计算到最近平面的距离
    print("\n计算噪声点到最近平面的距离...")
    noise_mask = df['is_noise'].values == 1
    noise_indices = np.where(noise_mask)[0]
    
    if len(noise_indices) > 0 and len(plane_dict) > 0:
        xyz_all = xyz_original
        noise_count = 0
        
        for noise_idx in noise_indices:
            noise_point = xyz_all[noise_idx]
            min_distance = np.inf
            nearest_plane_id = -1
            nearest_plane_str = None
            
            # ✅ 计算该噪声点到所有平面的距离，找最近的
            for plane_id, plane in plane_dict.items():
                if plane is not None:
                    dist = point_to_plane_distance(noise_point, plane)
                    if not np.isnan(dist) and dist < min_distance:
                        min_distance = dist
                        nearest_plane_id = plane_id
                        nearest_plane_str = f"{plane[0]:.6f},{plane[1]:.6f},{plane[2]:.6f},{plane[3]:.6f}"
            
            # ✅ 保存最近平面的距离和平面参数
            if min_distance < np.inf:
                distances[noise_idx] = min_distance
                plane_params_segmented[noise_idx] = nearest_plane_str
                noise_count += 1
        
        print(f"  ✓ 已处理 {noise_count} 个噪声点，成功计算距离")
    
    # 统计信息
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) > 0:
        print(f"\n✓ 距离计算完成")
        print(f"  保留点: {len(distances[keep_indices][~np.isnan(distances[keep_indices])])}")
        print(f"  噪声点: {len(distances[noise_indices][~np.isnan(distances[noise_indices])])}")
        print(f"  总有效距离数：{len(valid_distances)}")
        print(f"  最小距离: {np.min(valid_distances):.6f} m")
        print(f"  最大距离: {np.max(valid_distances):.6f} m")
        print(f"  平均距离: {np.mean(valid_distances):.6f} m")
        print(f"  中位数:   {np.median(valid_distances):.6f} m")
        print(f"  标准差:   {np.std(valid_distances):.6f} m")
    else:
        print("⚠ 警告：没有有效的距离数据")
    
    return distances, plane_params_segmented, plane_dict

def main():
    print("="*70)
    print("步骤02B：聚类合并局部平面")
    print("="*70)
    
    # ✅ 改动1：读取去噪后的数据（而不是原始数据）
    local_planes_path = 'script_find_xyz_points/result/02_local_planes.csv'
    
    print(f"\n读取步骤02A的结果: {local_planes_path}")
    if not os.path.exists(local_planes_path):
        print(f"❌ 错误：找不到文件 {local_planes_path}")
        print(f"请先运行步骤02A：python script_find_xyz_points/02_fit_local_planes.py")
        return
    
    df = pd.read_csv(local_planes_path)
    
    # ✅ 改动2：筛选保留点（而不是曲面点）
    keep_mask = df['is_noise'].values == 0
    keep_indices = np.where(keep_mask)[0]
    
    print(f"\n保留点统计：")
    print(f"  总点数：{len(df)}")
    print(f"  保留点：{len(keep_indices)} ({100*len(keep_indices)/len(df):.2f}%)")
    print(f"  噪声点：{len(df) - len(keep_indices)} ({100*(len(df) - len(keep_indices))/len(df):.2f}%)")
    
    # ✅ 改动3：只处理保留点
    xyz_surface = df.loc[keep_indices, ['x', 'y', 'z']].values.astype(np.float64)
    distances = df.loc[keep_indices, 'distance_to_local_plane'].values
    plane_params_str = df.loc[keep_indices, 'plane_params'].values
    
    # 解析平面参数
    plane_params = [parse_plane_params(s) for s in plane_params_str]
    
    # 归一化坐标
    scaler = StandardScaler()
    xyz_normalized = scaler.fit_transform(xyz_surface)
    
    # 构建KDTree
    print("\n构建KDTree索引...")
    kdtree = cKDTree(xyz_normalized)
    
    # 参数配置
    print("\n" + "="*70)
    print("聚类参数配置")
    print("="*70)
    
    xyz_range = np.max(xyz_normalized.max(axis=0) - xyz_normalized.min(axis=0))
    max_radius = xyz_range * 0.1
    normal_threshold = 0.8  # 法向量相似度阈值
    
    print(f"  最大邻域半径 (max_radius): {max_radius:.4f}")
    print(f"  法向量阈值 (normal_threshold): {normal_threshold}")
    
    # ✅ 贪心聚类
    print("\n" + "="*70)
    print("贪心聚类：将相似平面合并")
    print("="*70)
    
    plane_clusters = {}
    plane_id_counter = 0
    assigned = np.zeros(len(keep_indices), dtype=bool)
    
    for i in range(len(keep_indices)):
        if assigned[i]:
            continue
        
        # 创建新的平面簇
        current_plane_id = plane_id_counter
        plane_clusters[current_plane_id] = []
        
        current_plane = plane_params[i]
        if current_plane is None:
            assigned[i] = True
            continue
        
        # 获取邻域内的点
        query_point_normalized = xyz_normalized[i]
        neighbor_indices_list = kdtree.query_ball_point(query_point_normalized, max_radius)
        
        current_normal = np.array(current_plane[:3])
        current_normal = current_normal / np.linalg.norm(current_normal)
        
        # 将邻域内法向量相似的点分配给当前平面
        for neighbor_idx in neighbor_indices_list:
            if assigned[neighbor_idx]:
                continue
            
            neighbor_plane = plane_params[neighbor_idx]
            if neighbor_plane is None:
                continue
            
            neighbor_normal = np.array(neighbor_plane[:3])
            neighbor_normal = neighbor_normal / np.linalg.norm(neighbor_normal)
            
            dot_product = np.abs(np.dot(current_normal, neighbor_normal))
            
            if dot_product > normal_threshold:
                plane_clusters[current_plane_id].append(neighbor_idx)
                assigned[neighbor_idx] = True
        
        plane_id_counter += 1
    
    # 过滤小平面
    min_plane_size = 20
    valid_plane_clusters = {}
    valid_plane_id_counter = 0
    
    for plane_id, point_indices in plane_clusters.items():
        if len(point_indices) >= min_plane_size:
            valid_plane_clusters[valid_plane_id_counter] = point_indices
            valid_plane_id_counter += 1
    
    print(f"✓ 聚类结果：")
    print(f"  初始簇数：{plane_id_counter}")
    print(f"  有效平面数（点数≥{min_plane_size}）：{len(valid_plane_clusters)}")
    
    # ✅ 创建 plane_id 数组（映射到原始索引）
    plane_ids_full = np.full(len(df), -1, dtype=int)  # -1 表示噪声点或未分配
    
    plane_inliers = []
    plane_stats = []
    
    for plane_id, local_indices in valid_plane_clusters.items():
        local_indices = np.array(local_indices)
        original_indices = keep_indices[local_indices]  # 映射回原始索引
        plane_inliers.append(original_indices)
        
        # ✅ 为原始索引分配平面ID
        plane_ids_full[original_indices] = plane_id
        
        plane_distances = distances[local_indices]
        
        stats = {
            'plane_id': plane_id,
            'point_count': len(local_indices),
            'percentage': 100 * len(local_indices) / len(keep_indices),
            'avg_distance': np.mean(plane_distances),
            'max_distance': np.max(plane_distances),
            'min_distance': np.min(plane_distances)
        }
        plane_stats.append(stats)
    
    # 按点数降序排序
    plane_stats_sorted = sorted(plane_stats, key=lambda x: x['point_count'], reverse=True)
    
    print(f"\n  平面统计信息（按点数降序）：")
    print(f"  {'平面ID':<10} {'点数':<10} {'占比':<10} {'平均距离':<12} {'最大距离':<12} {'最小距离':<12}")
    print("-" * 76)
    
    for stat in plane_stats_sorted:
        print(f"  {stat['plane_id']:<10} {stat['point_count']:<10} {stat['percentage']:<10.2f}% " +
              f"{stat['avg_distance']:<12.6f} {stat['max_distance']:<12.6f} {stat['min_distance']:<12.6f}")
    
    # 保存聚类结果
    print("\n" + "="*70)
    print("保存结果...")
    print("="*70)
    
        # ✅ 计算距离到分割平面
    distances_to_segmented, plane_params_segmented, plane_dict = \
        calculate_distances_to_segmented_planes(
            df, 
            df[['x', 'y', 'z']].values.astype(np.float64),
            keep_indices,
            valid_plane_clusters,
            scaler
        )
    
    # ✅ 添加新列到DataFrame
    df['plane_id'] = plane_ids_full
    df['distance_to_segmented_plane'] = distances_to_segmented
    df['plane_params_segmented'] = plane_params_segmented
    
    output_path = 'script_find_xyz_points/result/03_segmented_planes.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ 结果已保存: {output_path}")
    print(f"✓ 已添加列:")
    print(f"    - plane_id：聚类平面ID")
    print(f"    - distance_to_segmented_plane：点到分割平面的距离（原始坐标系）")
    print(f"    - plane_params_segmented：聚类平面参数")
    
    # ✅ 在保存CSV时添加 plane_id 列
    df['plane_id'] = plane_ids_full
    
    output_path = 'script_find_xyz_points/result/03_segmented_planes.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ 结果已保存: {output_path}")
    print(f"✓ 已添加 'plane_id' 列")
    
    # ✅ 可视化方案1：全部点（包含噪声）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案1：全部点（包含噪声点，灰色显示）")
    print("="*70)
    visualize_all_planes_with_points(df[['x', 'y', 'z']].values, 
                                     plane_inliers,
                                     title="平面分割结果：各个聚类平面（含噪声点）")
    
    # ✅ 可视化方案2：仅保留点（不显示噪声）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案2：仅保留点（不显示噪声点）")
    print("="*70)
    visualize_segmented_planes_only_keep(df[['x', 'y', 'z']].values,
                                        df['is_noise'].values,
                                        plane_inliers,
                                        title="平面分割结果：各个聚类平面（仅保留点）")
    
    # 可视化方案3：距离热力图
    print("\n" + "="*70)
    print("3D点云可视化 - 方案3：距离热力图（仅保留点）")
    print("="*70)
    visualize_3d_point_cloud_continuous(
        xyz_surface,
        distances,
        title="保留点到局部平面的距离分布 - Viridis热力图",
        cmap_name="viridis",
        normalize_method="percentile"
    )
    
    print("\n" + "="*70)
    print("✓ 所有处理完成")
    print("="*70)


if __name__ == "__main__":
    main()