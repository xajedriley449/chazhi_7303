"""
步骤04：合并距离列并生成热力图
功能：读取02B的分割结果，合并 distance_to_local_plane 和 distance_to_segmented_plane，
      生成最终的 distance_to_plane 列，并进行热力图可视化
"""

import numpy as np
import pandas as pd
import os
import sys
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualize_3d_point_cloud_continuous, visualize_3d_point_cloud_with_distances, plot_xyzt_projections


def visualize_distance_heatmap_3d_rainbow_normalized(xyz, distances, title="点到平面距离热力图（离散彩虹色-归一化坐标）"):
    """3D可视化距离热力图 - 离散彩虹色映射（归一化坐标空间，凸显细节）"""
    print("\n" + "="*70)
    print(f"3D热力图可视化：{title}")
    print("="*70)
    
    xyz = np.asarray(xyz)
    distances = np.asarray(distances)
    
    # 提取有效距离
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) == 0:
        print("❌ 没有有效的距离数据")
        return
    
    # ✅ 归一化坐标到 [-1, 1] 范围
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_normalized = 2 * (xyz - xyz_min) / xyz_range - 1
    
    print(f"原始坐标范围:")
    print(f"  X: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}]")
    print(f"  Y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}]")
    print(f"  Z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")
    print(f"归一化坐标范围: [-1, 1]")
    
    min_dist = np.nanmin(distances)
    max_dist = np.nanmax(distances)
    
    print(f"距离范围: [{min_dist:.6f}, {max_dist:.6f}] m")
    
    if max_dist <= min_dist:
        print("⚠ 所有距离值相同，无法着色")
        normalized_distances = np.zeros_like(distances)
    else:
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    
    # 使用离散彩虹色映射
    cmap = cm.get_cmap('rainbow')
    colors = np.zeros((len(xyz_normalized), 3))
    
    for i in range(len(xyz_normalized)):
        if np.isnan(distances[i]):
            colors[i] = [0.5, 0.5, 0.5]  # 灰色：无效值
        else:
            rgba = cmap(normalized_distances[i])
            colors[i] = rgba[:3]
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_normalized)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_3d_point_cloud_continuous_normalized(xyz, distances, title: str = "3D点云距离分布（归一化坐标空间）", 
                                                   cmap_name: str = "viridis", normalize_method: str = "percentile"):
    """
    可视化3D点云（归一化坐标空间），支持多种正规化方法
    
    Args:
        xyz: 点的xyz坐标
        distances: 每个点的距离值
        title: 窗口标题
        cmap_name: colormap名称
        normalize_method: 正规化方法
    """
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # ✅ 归一化坐标到 [-1, 1] 范围
    xyz = np.asarray(xyz)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_normalized = 2 * (xyz - xyz_min) / xyz_range - 1
    
    print("\n" + "="*70)
    print(f"3D热力图可视化（归一化坐标空间）：{title}")
    print("="*70)
    print(f"原始坐标范围:")
    print(f"  X: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}]")
    print(f"  Y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}]")
    print(f"  Z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")
    print(f"归一化坐标范围: [-1, 1]")
    
    pcd.points = o3d.utility.Vector3dVector(xyz_normalized)
    
    distances = np.array(distances)
    valid_mask = distances < np.inf
    distances_valid = distances[valid_mask]
    
    if len(distances_valid) == 0:
        print("❌ 没有有效的距离数据")
        return
    
    dist_min = np.min(distances_valid)
    dist_max = np.max(distances_valid)
    dist_mean = np.mean(distances_valid)
    dist_median = np.median(distances_valid)
    
    print(f"距离统计:")
    print(f"  范围: [{dist_min:.2f}, {dist_max:.2f}]")
    print(f"  平均值: {dist_mean:.2f}, 中位数: {dist_median:.2f}")
    
    # 归一化距离
    distances_normalized = np.zeros_like(distances, dtype=float)
    
    if normalize_method == "percentile":
        percentile_low = np.percentile(distances_valid, 2)
        percentile_high = np.percentile(distances_valid, 98)
        print(f"  使用百分位数正规化: 2%-98% = [{percentile_low:.2f}, {percentile_high:.2f}]")
        
        if percentile_high > percentile_low:
            distances_normalized[valid_mask] = (distances[valid_mask] - percentile_low) / (percentile_high - percentile_low)
            distances_normalized[valid_mask] = np.clip(distances_normalized[valid_mask], 0, 1)
        else:
            distances_normalized[valid_mask] = 0.5
    
    elif normalize_method == "log":
        print(f"  使用对数正规化")
        log_distances = np.log1p(distances[valid_mask])
        log_min = np.min(log_distances)
        log_max = np.max(log_distances)
        
        if log_max > log_min:
            distances_normalized[valid_mask] = (log_distances - log_min) / (log_max - log_min)
        else:
            distances_normalized[valid_mask] = 0.5
    
    else:  # minmax
        print(f"  使用最小最大值正规化")
        if dist_max > dist_min:
            distances_normalized[valid_mask] = (distances[valid_mask] - dist_min) / (dist_max - dist_min)
        else:
            distances_normalized[valid_mask] = 0.5
    
    distances_normalized[~valid_mask] = -0.1
    
    # 使用连续色映射
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(distances_normalized)[:, :3]
    colors[~valid_mask] = [0.5, 0.5, 0.5]  # 无效点灰色
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 显示点云
    print(f"生成3D点云可视化（{normalize_method}正规化，归一化坐标）...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{title}\n范围={dist_min:.2f}~{dist_max:.2f}m | 中位数={dist_median:.2f}m\n({normalize_method}正规化，归一化坐标)",
        width=1200,
        height=800
    )


def export_distance_heatmap_pcd(xyz, distances, output_path):
    """导出距离热力图为PCD文件"""
    print(f"\n导出距离热力图PCD: {output_path}")
    
    xyz = np.asarray(xyz)
    distances = np.asarray(distances)
    
    # 提取有效距离
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) == 0:
        print("❌ 没有有效的距离数据")
        return
    
    # 归一化距离
    min_dist = np.nanmin(distances)
    max_dist = np.nanmax(distances)
    
    if max_dist <= min_dist:
        normalized_distances = np.zeros_like(distances)
    else:
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    
    # 使用 viridis 颜色映射
    cmap = cm.get_cmap('viridis')
    colors = np.zeros((len(xyz), 3))
    
    for i in range(len(xyz)):
        if np.isnan(distances[i]):
            colors[i] = [0.5, 0.5, 0.5]
        else:
            rgba = cmap(normalized_distances[i])
            colors[i] = rgba[:3]
    
    # 创建点云并保存
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print(f"✓ 已保存: {output_path}")


def plot_distance_distribution(distances, output_path):
    """绘制距离分布直方图和统计信息"""
    print(f"\n绘制距离分布直方图: {output_path}")
    
    valid_distances = distances[~np.isnan(distances)]
    
    if len(valid_distances) == 0:
        print("❌ 没有有效的距离数据")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('点到分割平面的距离分析', fontsize=16, fontweight='bold')
    
    # 1. 直方图
    ax = axes[0, 0]
    ax.hist(valid_distances, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('点数')
    ax.set_title('距离分布直方图')
    ax.grid(True, alpha=0.3)
    
    # 2. 累积分布
    ax = axes[0, 1]
    sorted_distances = np.sort(valid_distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    ax.plot(sorted_distances, cumulative, linewidth=2, color='darkblue')
    ax.set_xlabel('距离 (m)')
    ax.set_ylabel('累积比例')
    ax.set_title('累积分布函数')
    ax.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax = axes[1, 0]
    bp = ax.boxplot(valid_distances, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('距离 (m)')
    ax.set_title('距离箱线图')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 统计信息表
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    统计信息：
    
    点数：{len(valid_distances)}
    
    最小值：{np.min(valid_distances):.6f} m
    最大值：{np.max(valid_distances):.6f} m
    平均值：{np.mean(valid_distances):.6f} m
    中位数：{np.median(valid_distances):.6f} m
    标准差：{np.std(valid_distances):.6f} m
    
    25百分位：{np.percentile(valid_distances, 25):.6f} m
    75百分位：{np.percentile(valid_distances, 75):.6f} m
    
    95百分位：{np.percentile(valid_distances, 95):.6f} m
    99百分位：{np.percentile(valid_distances, 99):.6f} m
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    plt.close()


def main():
    print("="*70)
    print("步骤04：合并距离列并生成热力图")
    print("="*70)
    
    # 读取02B的分割结果
    input_path = 'script_find_xyz_points/result/03_segmented_planes.csv'
    
    print(f"\n读取分割结果: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到文件 {input_path}")
        print(f"请先运行步骤02B：python script_find_xyz_points/03_cluster_planes.py")
        return
    
    df = pd.read_csv(input_path)
    print(f"✓ 加载 {len(df)} 个点")
    
    # ✅ 合并距离列
    print("\n" + "="*70)
    print("合并距离列")
    print("="*70)
    
    if 'distance_to_local_plane' not in df.columns:
        print("❌ 错误：缺少 'distance_to_local_plane' 列")
        return
    
    if 'distance_to_segmented_plane' not in df.columns:
        print("❌ 错误：缺少 'distance_to_segmented_plane' 列")
        return
    
    # ✅ 创建最终的 distance_to_plane 列
    df['distance_to_plane'] = np.where(
        df['is_noise'] == 1,
        df['distance_to_segmented_plane'],
        df['distance_to_local_plane']
    )
    
    # 统计信息
    keep_distances = df.loc[df['is_noise'] == 0, 'distance_to_local_plane'].dropna()
    noise_distances = df.loc[df['is_noise'] == 1, 'distance_to_segmented_plane'].dropna()
    all_distances = df['distance_to_plane'].dropna()
    
    print(f"✓ 已合并距离列")
    print(f"\n  保留点（distance_to_local_plane）:")
    print(f"    点数: {len(keep_distances)}")
    if len(keep_distances) > 0:
        print(f"    范围: [{np.min(keep_distances):.6f}, {np.max(keep_distances):.6f}] m")
        print(f"    平均: {np.mean(keep_distances):.6f} m")
    
    print(f"\n  噪声点（distance_to_segmented_plane）:")
    print(f"    点数: {len(noise_distances)}")
    if len(noise_distances) > 0:
        print(f"    范围: [{np.min(noise_distances):.6f}, {np.max(noise_distances):.6f}] m")
        print(f"    平均: {np.mean(noise_distances):.6f} m")
    
    print(f"\n  合计（distance_to_plane）:")
    print(f"    点数: {len(all_distances)}")
    if len(all_distances) > 0:
        print(f"    范围: [{np.min(all_distances):.6f}, {np.max(all_distances):.6f}] m")
        print(f"    平均: {np.mean(all_distances):.6f} m")
    
    # 保存结果
    print("\n" + "="*70)
    print("保存结果...")
    print("="*70)
    
    # ✅ 只保留需要的列
    output_columns = ['x', 'y', 'z', 'e', 't', 'distance_to_plane']
    df_output = df[output_columns]
    
    #output_csv_path = 'script_calculate_distance/result/04_distance_calculated.csv'
    output_csv_path = 'script_calculate_distance/result/xyz_point_distances.csv'
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_output.to_csv(output_csv_path, index=False)
    print(f"✓ 已保存: {output_csv_path}")
    print(f"  包含列: {', '.join(df.columns.tolist())}")
    print(f"  行数: {len(df_output)}")
    
    # 提取坐标和距离用于可视化
    xyz = df[['x', 'y', 'z']].values
    distances = df['distance_to_plane'].values
    
    # ✅ 第一个3D窗口：离散彩虹色（原始坐标，调用utils）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案1：离散彩虹色（原始坐标空间，与2D投影配色一致）⭐")
    print("="*70)
    visualize_3d_point_cloud_with_distances(
        xyz, distances,
        title="点到平面的距离分布 - 离散彩虹色（原始坐标）"
    )
    
    # ✅ 第二个3D窗口：离散彩虹色（归一化坐标，自定义）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案2：离散彩虹色（归一化坐标空间，凸显细节）⭐")
    print("="*70)
    visualize_distance_heatmap_3d_rainbow_normalized(
        xyz, distances,
        title="点到平面的距离分布 - 离散彩虹色（归一化坐标，凸显细节）"
    )
    
    # ✅ 第三个3D窗口：连续Viridis色映射（原始坐标，调用utils）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案3：连续Viridis色映射（原始坐标空间）⭐")
    print("="*70)
    visualize_3d_point_cloud_continuous(
        xyz, distances,
        title="点到平面的距离分布 - 连续Viridis色映射（原始坐标）",
        cmap_name="viridis",
        normalize_method="percentile"
    )
    
    # ✅ 第四个3D窗口：连续Viridis色映射（归一化坐标，自定义）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案4：连续Viridis色映射（归一化坐标空间，凸显细节）⭐")
    print("="*70)
    visualize_3d_point_cloud_continuous_normalized(
        xyz, distances,
        title="点到平面的距离分布 - 连续Viridis色映射（归一化坐标，凸显细节）",
        cmap_name="viridis",
        normalize_method="percentile"
    )
    
    # ✅ 第五个窗口：2D投影可视化（调用utils）
    print("\n" + "="*70)
    print("2D投影可视化 - 方案5：XYZ三维投影（离散彩虹色）⭐")
    print("="*70)
    fig, point_colors = plot_xyzt_projections(
        df,
        labels=distances
    )
    plt.suptitle('点到分割平面的距离分布 (2D投影 - 离散彩虹色)', fontsize=14, fontweight='bold')
    plt.show()
    
    # ✅ 导出PCD文件
    pcd_path = 'script_calculate_distance/result/04_distance_heatmap.pcd'
    #export_distance_heatmap_pcd(xyz, distances, pcd_path)
    
    # ✅ 绘制分布直方图
    plot_path = 'script_calculate_distance/result/04_distance_distribution.png'
    #plot_distance_distribution(distances, plot_path)
    
    # 汇总
    print("\n" + "="*70)
    print("✓ 步骤04完成")
    print("="*70)
    print("\n生成的文件：")
    print(f"  1. {output_csv_path}")
    print(f"     └─ 包含所有点的 distance_to_plane 列")
    print(f"  2. {pcd_path}")
    print(f"     └─ 可用CloudCompare等工具打开")
    print(f"  3. {plot_path}")
    print(f"     └─ 距离分布统计图表")
    print("\n可视化窗口顺序（共5个）：")
    print(f"  1. 方案1：3D - 离散彩虹色（原始坐标空间，与2D投影配色一致）[utils实现] ⭐")
    print(f"  2. 方案2：3D - 离散彩虹色（归一化坐标空间 [-1,1]，凸显细节）[自定义] ⭐")
    print(f"  3. 方案3：3D - 连续Viridis色映射（原始坐标空间）[utils实现] ⭐")
    print(f"  4. 方案4：3D - 连续Viridis色映射（归一化坐标空间 [-1,1]，凸显细节）[自定义] ⭐")
    print(f"  5. 方案5：2D - XYZ三维投影（离散彩虹色）[utils实现] ⭐")


if __name__ == "__main__":
    main()