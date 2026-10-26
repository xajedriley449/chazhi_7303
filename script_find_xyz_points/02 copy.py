import numpy as np
import pandas as pd
import os
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

import sys
import open3d as o3d
import colorsys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 全局绘图参数（对齐论文排版风格）
# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, # 解决负号显示问题
    'font.size': 12,               
    'axes.labelsize': 18,          
    'xtick.labelsize': 16,         
    'ytick.labelsize': 16,         
    'svg.fonttype': 'path',        # 文字转轮廓
})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import visualize_3d_point_cloud_continuous

def fit_plane_least_squares(points):
    """使用最小二乘法拟合平面（SVD方法）"""
    if len(points) < 3:
        return None, None
    
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    U, S, Vt = np.linalg.svd(points_centered)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)
    
    a, b, c = normal
    d = -np.dot(normal, centroid)
    distances = np.abs(np.dot(points, normal) + d)
    
    return (a, b, c, d), distances


def point_to_plane_distance(point, plane):
    """计算点到平面的距离"""
    a, b, c, d = plane
    x, y, z = point
    distance = abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)
    return distance


def fit_plane_with_expanding_sphere(point_index, xyz, kdtree, 
                                    min_points=50, max_radius=None,
                                    initial_radius=10.0, radius_step=5.0,
                                    normals=None, normal_threshold=0.7):
    """对单个点执行动态球体扩展法拟合平面"""
    query_point = xyz[point_index]
    
    if max_radius is None:
        max_radius = initial_radius * 10
    
    current_radius = initial_radius
    best_plane = None
    best_distance = np.inf
    best_inlier_count = 0
    used_radius = initial_radius
    
    while current_radius <= max_radius:
        neighbor_indices = kdtree.query_ball_point(query_point, current_radius)
        neighbor_count = len(neighbor_indices)
        
        if neighbor_count >= min_points:
            neighbor_points = xyz[neighbor_indices]
            plane, distances = fit_plane_least_squares(neighbor_points)
            
            if plane is not None:
                if normals is not None:
                    neighbor_indices_array = np.array(neighbor_indices)
                    inlier_normals = normals[neighbor_indices_array]
                    plane_normal = np.array(plane[:3])
                    plane_normal = plane_normal / np.linalg.norm(plane_normal)
                    dot_products = np.abs(np.dot(inlier_normals, plane_normal))
                    normal_consistent = dot_products > normal_threshold
                    
                    if np.sum(normal_consistent) >= min_points:
                        neighbor_indices_array = neighbor_indices_array[normal_consistent]
                        neighbor_points = xyz[neighbor_indices_array]
                        plane, distances = fit_plane_least_squares(neighbor_points)
                
                best_plane = plane
                best_distance = point_to_plane_distance(query_point, plane)
                best_inlier_count = len(neighbor_indices_array) if normals is not None else neighbor_count
                used_radius = current_radius
                break
        
        current_radius += radius_step
    
    if best_plane is None:
        neighbor_indices = kdtree.query_ball_point(query_point, min(current_radius, max_radius))
        if len(neighbor_indices) >= 3:
            neighbor_points = xyz[neighbor_indices]
            plane, distances = fit_plane_least_squares(neighbor_points)
            if plane is not None:
                best_plane = plane
                best_distance = point_to_plane_distance(query_point, plane)
                best_inlier_count = len(neighbor_indices)
                used_radius = min(current_radius, max_radius)
    
    return best_plane, best_distance, best_inlier_count, used_radius

def visualize_local_planes_before_clustering(xyz, plane_params_str_list, title="局部平面拟合结果（聚类前）"):
    """用不同颜色显示不同朝向的局部平面"""
    print("\n" + "="*70)
    print(f"可视化：{title}")
    print("="*70)
    
    geometries = []
    
    # 解析平面参数字符串为元组
    plane_params = []
    for p_str in plane_params_str_list:
        if p_str == "None" or pd.isna(p_str):
            plane_params.append(None)
        else:
            try:
                parts = [float(x) for x in p_str.split(',')]
                plane_params.append(tuple(parts))
            except:
                plane_params.append(None)
    
    # 提取法向量并按方向聚类着色
    normals_array = []
    for p in plane_params:
        if p is not None and len(p) >= 3:
            normal = np.array(p[:3])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals_array.append(normal / norm)
            else:
                normals_array.append(np.array([0, 0, 1]))
        else:
            normals_array.append(np.array([0, 0, 1]))
    
    normals_array = np.array(normals_array)
    
    # 用法向量的HSV颜色映射
    xyz = np.asarray(xyz)
    colors = np.zeros((len(xyz), 3))
    
    for i in range(len(xyz)):
        if plane_params[i] is not None:
            normal = normals_array[i]
            # 把法向量转换为HSV色彩
            hue = (np.arctan2(normal[1], normal[0]) + np.pi) / (2 * np.pi)  # 0-1
            saturation = np.sqrt(normal[0]**2 + normal[1]**2)  # 水平分量
            value = np.clip(normal[2] + 0.5, 0, 1)  # Z分量映射到亮度
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[i] = [r, g, b]
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色表示拟合失败
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries, window_name=title)


def visualize_local_planes_only_keep(xyz_all, is_noise, plane_params_str_list, 
                                     title="局部平面拟合结果（仅保留点）"):
    """✅ 新增：仅显示保留点的局部平面拟合结果（不显示噪声点）"""
    print("\n" + "="*70)
    print(f"可视化（仅保留点）：{title}")
    print("="*70)
    
    # 筛选保留点
    keep_mask = is_noise == 0
    keep_indices = np.where(keep_mask)[0]
    
    xyz_keep = np.asarray(xyz_all)[keep_indices]
    plane_params_keep = [plane_params_str_list[i] for i in keep_indices]
    
    # 解析平面参数字符串为元组
    plane_params = []
    for p_str in plane_params_keep:
        if p_str == "None" or pd.isna(p_str):
            plane_params.append(None)
        else:
            try:
                parts = [float(x) for x in p_str.split(',')]
                plane_params.append(tuple(parts))
            except:
                plane_params.append(None)
    
    # 提取法向量并按方向着色
    normals_array = []
    for p in plane_params:
        if p is not None and len(p) >= 3:
            normal = np.array(p[:3])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals_array.append(normal / norm)
            else:
                normals_array.append(np.array([0, 0, 1]))
        else:
            normals_array.append(np.array([0, 0, 1]))
    
    normals_array = np.array(normals_array)
    
    # 用法向量的HSV颜色映射
    colors = np.zeros((len(xyz_keep), 3))
    
    for i in range(len(xyz_keep)):
        if plane_params[i] is not None:
            normal = normals_array[i]
            # 把法向量转换为HSV色彩
            hue = (np.arctan2(normal[1], normal[0]) + np.pi) / (2 * np.pi)  # 0-1
            saturation = np.sqrt(normal[0]**2 + normal[1]**2)  # 水平分量
            value = np.clip(normal[2] + 0.5, 0, 1)  # Z分量映射到亮度
            
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors[i] = [r, g, b]
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色表示拟合失败
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_keep)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd], window_name=title)

def visualize_matplotlib_style(df, result_dir):
    """
    生成归一化风格的多视图投影（根据平面法向量进行多彩HSV着色，噪声点为灰）
    """
    img_dir = os.path.join(result_dir, '02_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 空间坐标归一化
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(df[['x']]).flatten()
    y_norm = scaler.fit_transform(df[['y']]).flatten()
    z_norm = scaler.fit_transform(df[['z']]).flatten()
    
    # 从 df 中提取颜色（复用 HSV 法向量着色逻辑）
    colors = np.zeros((len(df), 3))
    
    for i, row in df.iterrows():
        p_str = row['plane_params']
        is_noise = row['is_noise']
        
        # 针对空值或标记为噪声的点赋灰色
        if pd.isna(p_str) or str(p_str).strip() == "None" or is_noise == 1:
            colors[i] = [0.5, 0.5, 0.5]
        else:
            try:
                parts = [float(x) for x in str(p_str).split(',')]
                if len(parts) >= 3:
                    normal = np.array(parts[:3])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-6:
                        normal = normal / norm
                    else:
                        normal = np.array([0, 0, 1])
                        
                    # 计算 HSV
                    hue = (np.arctan2(normal[1], normal[0]) + np.pi) / (2 * np.pi) 
                    saturation = np.sqrt(normal[0]**2 + normal[1]**2) 
                    value = np.clip(normal[2] + 0.5, 0, 1) 
                    
                    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                    colors[i] = [r, g, b]
                else:
                    colors[i] = [0.5, 0.5, 0.5]
            except:
                colors[i] = [0.5, 0.5, 0.5]
                
    print(f"\n生成 Matplotlib 风格投影图 (彩色=拟合成功的法向平面，灰=噪声点/失败点)...")
    
    # ================= 1. 三维视图 (X-Y-Z) =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_norm, y_norm, z_norm, c=colors, s=2, alpha=0.6)
    
    # 【修改】：使用 mathtext 格式化坐标轴标签
    ax.set_xlabel(r'$\mathrm{X}$', labelpad=15)
    ax.set_ylabel(r'$\mathrm{Y}$', labelpad=15)
    ax.set_zlabel(r'$\mathrm{Z}$', labelpad=10)
    
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    ax.tick_params(axis='z', pad=8)
    
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-45)  # 经典视角
    
    # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
    formatter = plt.ScalarFormatter(useOffset=False)
    formatter.set_useMathText(True)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=12)
    
    # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
    fig.canvas.draw()
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    for label in tick_labels:
        label.set_fontname('Times New Roman')
    
    save_path_png = os.path.join(img_dir, 'XYZ_3D_projection.png')
    save_path_svg = os.path.join(img_dir, 'XYZ_3D_projection.svg')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"已保存: {save_path_png}")

    # ================= 2. 二维视图 (XY, XZ, YZ) =================
    # 【修改】：使用 mathtext 格式化坐标轴标签
    projections = [
        ('XY', x_norm, y_norm, r'$\mathrm{X}$', r'$\mathrm{Y}$'),
        ('XZ', x_norm, z_norm, r'$\mathrm{X}$', r'$\mathrm{Z}$'),
        ('YZ', y_norm, z_norm, r'$\mathrm{Y}$', r'$\mathrm{Z}$')
    ]
    
    for title, px, py, xlabel, ylabel in projections:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(px, py, c=colors, s=2, alpha=0.6)
        
        ax.set_xlabel(f'{xlabel}', labelpad=5)
        ax.set_ylabel(f'{ylabel}', labelpad=5)
        
        # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.tick_params(axis='both', labelsize=14)
        
        # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
        fig.canvas.draw()
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in tick_labels:
            label.set_fontname('Times New Roman')
            
        plt.tight_layout()
        save_path_png = os.path.join(img_dir, f'{title}_projection.png')
        save_path_svg = os.path.join(img_dir, f'{title}_projection.svg')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path_png}")
    
def denormalize_distance(normalized_distance, scaler):
    """将归一化坐标系中的距离转换为原始坐标系的距离"""
    if np.isnan(normalized_distance) or normalized_distance == np.inf:
        return normalized_distance
    
    # ❌ 错误的做法：除以几何均值会让距离变小
    # scale_geometric_mean = np.power(np.prod(scaler.scale_), 1.0/3.0)
    # return normalized_distance / scale_geometric_mean
    
    # ✅ 正确的做法：乘以几何均值
    scale_geometric_mean = np.power(np.prod(scaler.scale_), 1.0/3.0)
    return normalized_distance * scale_geometric_mean  # 改为乘法！



def main():
    print("="*70)
    print("步骤02A：对保留点拟合局部平面（基于去噪结果）")
    print("="*70)
    
    # ✅ 改动1：读取去噪后的数据
    denoised_path = 'script_find_xyz_points/result/01_denoised_points.csv'
    
    print(f"\n读取去噪数据: {denoised_path}")
    if not os.path.exists(denoised_path):
        print(f"❌ 错误：找不到文件 {denoised_path}")
        return
    
    df = pd.read_csv(denoised_path)
    
    # ✅ 改动2：筛选保留点（去噪标记为0）
    keep_mask = df['is_noise'].values == 0
    keep_indices = np.where(keep_mask)[0]
    
    print(f"\n保留点统计：")
    print(f"  总点数：{len(df)}")
    print(f"  保留点：{len(keep_indices)} ({100*len(keep_indices)/len(df):.2f}%)")
    print(f"  噪声点：{len(df) - len(keep_indices)} ({100*(len(df) - len(keep_indices))/len(df):.2f}%)")
    
    # ✅ 改动3：提取保留点数据
    xyz_keep = df.loc[keep_indices, ['x', 'y', 'z']].values.astype(np.float64)
    
    print(f"\n保留点坐标范围：")
    print(f"  X: [{xyz_keep[:, 0].min():.2f}, {xyz_keep[:, 0].max():.2f}]")
    print(f"  Y: [{xyz_keep[:, 1].min():.2f}, {xyz_keep[:, 1].max():.2f}]")
    print(f"  Z: [{xyz_keep[:, 2].min():.2f}, {xyz_keep[:, 2].max():.2f}]")
    
    # ✅ 改动4：计算法向量（使用Open3D）
    print("\n" + "="*70)
    print("计算法向量...")
    print("="*70)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_keep)
    
    # 自动计算搜索半径
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = avg_dist * 5
    
    print(f"  平均点间距: {avg_dist:.6f}")
    print(f"  搜索半径: {radius:.6f}")
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    normals_keep = np.asarray(pcd.normals)
    print(f"✓ 法向量计算完成")
    
    # 归一化坐标
    scaler = StandardScaler()
    xyz_normalized = scaler.fit_transform(xyz_keep)
    
    print(f"\n归一化后坐标范围：")
    print(f"  X: [{xyz_normalized[:, 0].min():.4f}, {xyz_normalized[:, 0].max():.4f}]")
    print(f"  Y: [{xyz_normalized[:, 1].min():.4f}, {xyz_normalized[:, 1].max():.4f}]")
    print(f"  Z: [{xyz_normalized[:, 2].min():.4f}, {xyz_normalized[:, 2].max():.4f}]")
    
    xyz_range = np.max(xyz_normalized.max(axis=0) - xyz_normalized.min(axis=0))
    
    # 构建KDTree
    print("\n" + "="*70)
    print("构建KDTree索引...")
    print("="*70)
    kdtree = cKDTree(xyz_normalized)
    print("✓ KDTree构建完成")
    
    # 参数配置
    print("\n" + "="*70)
    print("参数配置")
    print("="*70)
    
    min_points = 30
    initial_radius = xyz_range * 0.015
    radius_step = xyz_range * 0.005
    max_radius = xyz_range * 0.1
    normal_threshold = 0.8
    
    print(f"  最少邻域点数 (min_points): {min_points}")
    print(f"  初始球半径 (initial_radius): {initial_radius:.4f}")
    print(f"  扩展步长 (radius_step): {radius_step:.4f}")
    print(f"  最大球半径 (max_radius): {max_radius:.4f}")
    print(f"  法向量阈值 (normal_threshold): {normal_threshold}")
    
    # 对每个保留点执行球体扩展法
    print("\n" + "="*70)
    print("对每个保留点执行球体扩展法...")
    print("="*70)
    
    distances = []
    inlier_counts = []
    used_radii = []
    plane_params_list = []
    
    for i, point_idx in enumerate(keep_indices):
        plane, distance, inlier_count, used_radius = fit_plane_with_expanding_sphere(
            point_index=i,
            xyz=xyz_normalized,
            kdtree=kdtree,
            min_points=min_points,
            max_radius=max_radius,
            initial_radius=initial_radius,
            radius_step=radius_step,
            normals=normals_keep,
            normal_threshold=normal_threshold
        )
        
        distances.append(distance)
        inlier_counts.append(inlier_count)
        used_radii.append(used_radius)
        
        # 保存平面参数 (a,b,c,d)
        if plane is not None:
            plane_params_list.append(f"{plane[0]:.6f},{plane[1]:.6f},{plane[2]:.6f},{plane[3]:.6f}")
        else:
            plane_params_list.append("None")
        
        if (i + 1) % 500 == 0:
            print(f"  已处理: {i + 1:6d} / {len(keep_indices)}")
    
    distances = np.array(distances)
    inlier_counts = np.array(inlier_counts)
    used_radii = np.array(used_radii)
    
    print(f"  已处理: {len(keep_indices):6d} / {len(keep_indices)} ✓")
    
    # ✅ 改动5：创建完整的平面参数列表（包括噪声点）
    plane_params_list_full = ["None"] * len(df)
    for idx, param in zip(keep_indices, plane_params_list):
        plane_params_list_full[idx] = param
    
    # 保存结果
    print("\n" + "="*70)
    print("保存结果...")
    print("="*70)
    
    result_df = df.copy()
    result_df['distance_to_local_plane'] = np.nan
    result_df['local_inlier_count'] = 0
    result_df['local_sphere_radius'] = 0.0
    result_df['plane_params'] = ''
    
    # ✅ 关键改动：保存前反归一化距离
    distances_denormalized = np.array([
        denormalize_distance(d, scaler) for d in distances
    ])
    
    #result_df.loc[keep_indices, 'distance_to_local_plane'] = distances
    result_df.loc[keep_indices, 'distance_to_local_plane'] = distances_denormalized
    result_df.loc[keep_indices, 'local_inlier_count'] = inlier_counts
    result_df.loc[keep_indices, 'local_sphere_radius'] = used_radii
    result_df.loc[keep_indices, 'plane_params'] = plane_params_list
    
    os.makedirs('script_find_xyz_points/result', exist_ok=True)
    output_path = 'script_find_xyz_points/result/02_local_planes.csv'
    result_df.to_csv(output_path, index=False)
    print(f"✓ 结果已保存: {output_path}")
    
    # 统计信息
    print("\n" + "="*70)
    print("距离统计信息")
    print("="*70)
    
    valid_distances = distances[distances < np.inf]
    
    if len(valid_distances) > 0:
        print(f"  有效距离点数: {len(valid_distances)} / {len(distances)}")
        print(f"  最小值:       {np.min(valid_distances):10.6f} m")
        print(f"  最大值:       {np.max(valid_distances):10.6f} m")
        print(f"  平均值:       {np.mean(valid_distances):10.6f} m")
        print(f"  中位数:       {np.median(valid_distances):10.6f} m")
        print(f"  标准差:       {np.std(valid_distances):10.6f} m")
        print(f"  25百分位:     {np.percentile(valid_distances, 25):10.6f} m")
        print(f"  75百分位:     {np.percentile(valid_distances, 75):10.6f} m")
        
        print(f"\n邻域点数统计:")
        print(f"  最小值:       {np.min(inlier_counts):6d} 个点")
        print(f"  最大值:       {np.max(inlier_counts):6d} 个点")
        print(f"  平均值:       {np.mean(inlier_counts):6.1f} 个点")
        print(f"  中位数:       {np.median(inlier_counts):6.1f} 个点")
    
    print("\n" + "="*70)
    print("✓ 步骤02A完成")
    print("="*70)
    
    # ✅ 可视化方案1：全部点（包含噪声）
    print("\n" + "="*70)
    print("可视化方案1：全部点（包含噪声点，灰色显示）")
    print("="*70)
    visualize_local_planes_before_clustering(
        df[['x', 'y', 'z']].values,
        plane_params_list_full,  # ✅ 使用完整列表
        title="局部平面拟合结果：法向量着色（含噪声点）"
    )
    
    # ✅ 可视化方案2：仅保留点（不显示噪声）
    print("\n" + "="*70)
    print("可视化方案2：仅保留点（不显示噪声点）")
    print("="*70)
    visualize_local_planes_only_keep(
        df[['x', 'y', 'z']].values,
        df['is_noise'].values,
        plane_params_list_full,  # ✅ 使用完整列表
        title="局部平面拟合结果：法向量着色（仅保留点）"
    )

    # ✅ 可视化方案3：生成 Matplotlib 论文标准风格的 2D/3D 高清截图导出
    visualize_matplotlib_style(result_df, 'script_find_xyz_points/result')
    print("🎉 局部平面拟合图表导出完成！")

if __name__ == "__main__":
    main()
