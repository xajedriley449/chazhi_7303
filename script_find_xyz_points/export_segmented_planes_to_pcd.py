import numpy as np
import pandas as pd
import os
import open3d as o3d
import colorsys


def export_segmented_planes_to_pcd(csv_path, output_dir='script_find_xyz_points/result/pcd_output'):
    """将平面分割结果导出为PCD文件"""
    print("="*70)
    print("步骤03：导出平面分割结果为PCD格式")
    print("="*70)
    
    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到文件 {csv_path}")
        return
    
    print(f"\n读取分割结果: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ 加载 {len(df)} 个点")
    
    points = df[['x', 'y', 'z']].values.astype(np.float64)
    os.makedirs(output_dir, exist_ok=True)
    
    # ✅ 方案1：按平面ID着色
    print("\n" + "="*70)
    print("方案1：按平面ID着色（含噪声点）")
    print("="*70)
    
    plane_ids = df['plane_id'].values
    valid_plane_ids = np.unique(plane_ids[plane_ids >= 0])
    num_planes = len(valid_plane_ids)
    
    print(f"有效平面数: {num_planes}")
    
    colors_plane = np.zeros((len(points), 3))
    
    for rank, pid in enumerate(sorted(valid_plane_ids)):
        hue = (rank / max(num_planes, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors_plane[plane_ids == pid] = [r, g, b]
        print(f"  平面 {pid}: RGB({r:.2f}, {g:.2f}, {b:.2f})")
    
    # 噪声点灰色
    colors_plane[plane_ids < 0] = [0.65, 0.65, 0.65]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_plane)
    
    output_file = os.path.join(output_dir, '02B_segmented_planes_colored.pcd')
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"✓ 已保存: {output_file}")
    
    # ✅ 方案2：按噪声标记着色
    print("\n" + "="*70)
    print("方案2：按噪声标记着色")
    print("="*70)
    
    colors_noise = np.where(
        df['is_noise'].values[:, None] == 0,
        np.array([0.0, 1.0, 0.0]),  # 绿色：保留点
        np.array([1.0, 0.0, 0.0])   # 红色：噪声点
    )
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_noise)
    
    output_file = os.path.join(output_dir, '02B_noise_classification.pcd')
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"✓ 已保存: {output_file}")
    
    # ✅ 方案3：按距离热力图
    print("\n" + "="*70)
    print("方案3：按距离热力图")
    print("="*70)
    
    distances = df['distance_to_local_plane'].values
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) > 0:
        min_dist = np.nanmin(distances)
        max_dist = np.nanmax(distances)
        normalized = (distances - min_dist) / (max_dist - min_dist + 1e-6)
    else:
        normalized = np.zeros_like(distances)
    
    # Viridis 热力图
    colors_heat = np.zeros((len(points), 3))
    for i, t in enumerate(normalized):
        if np.isnan(t):
            colors_heat[i] = [0.5, 0.5, 0.5]
        elif t < 0.25:
            colors_heat[i] = [0.267, 0.0, 0.329 - t/0.25*0.329]
        elif t < 0.5:
            colors_heat[i] = [0.267, (t-0.25)/0.25*0.560, 0.329]
        elif t < 0.75:
            colors_heat[i] = [0.121 + (t-0.5)/0.25*0.82, 0.560 + (t-0.5)/0.25*0.415, 0.0]
        else:
            colors_heat[i] = [0.941 + (t-0.75)/0.25*0.059, 0.975 + (t-0.75)/0.25*0.025, 0.131 + (t-0.75)/0.25*0.869]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_heat)
    
    output_file = os.path.join(output_dir, '02B_distance_heatmap.pcd')
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    print(f"✓ 已保存: {output_file}")
    print(f"距离范围: [{min_dist:.6f}, {max_dist:.6f}] m")
    
    print("\n" + "="*70)
    print("✓ PCD导出完成")
    print("="*70)
    print(f"\n输出目录: {output_dir}")
    print("生成文件：")
    print("  1. 02B_segmented_planes_colored.pcd - 按平面ID着色（推荐）")
    print("  2. 02B_noise_classification.pcd     - 按噪声标记")
    print("  3. 02B_distance_heatmap.pcd         - 距离热力图")


if __name__ == "__main__":
    csv_path = 'script_find_xyz_points/result/02B_segmented_planes.csv'
    export_segmented_planes_to_pcd(csv_path)