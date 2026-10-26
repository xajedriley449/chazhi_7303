from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_xyzt_projections, visualize_3d_point_cloud_with_distances, visualize_3d_point_cloud_continuous

# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False, # 解决负号显示问题
    # 【致胜修改】：全面提高所有元素的基准字号
    'font.size': 16,               # 基础字号从 12 提升到 14
    'axes.titlesize': 20,          # 标题字号 18
    'axes.labelsize': 18,          # 坐标轴标签字号 16
    'xtick.labelsize': 16,         # 刻度数字 14
    'ytick.labelsize': 16,
    'svg.fonttype': 'path',        # 文字转轮廓，保证中文显示
})

def visualize_matplotlib_style(df, result_dir, normalize_method="percentile"):
    """
    生成归一化多视图投影（基于distance_to_nearest_line值着色）
    
    Args:
        df: 包含x, y, z, t, distance_to_nearest_line列的DataFrame
        result_dir: 结果保存目录
        normalize_method: 归一化方法
            - "percentile": 百分位数正规化（推荐，忽略极值）
            - "log": 对数正规化（适合跨度大的数据）
            - "minmax": 最小最大值正规化（原始方法）
    """
    img_dir = os.path.join(result_dir, 't_img')
    os.makedirs(img_dir, exist_ok=True)
    
    # 坐标系空间及时间归一化可视化
    scaler = MinMaxScaler()
    x_norm = scaler.fit_transform(df[['x']]).flatten()
    y_norm = scaler.fit_transform(df[['y']]).flatten()
    z_norm = scaler.fit_transform(df[['z']]).flatten()
    t_norm = scaler.fit_transform(df[['t']]).flatten()  # 提取并归一化时间 T
    
    # 使用distance_to_nearest_line作为颜色值
    distances = df['distance_to_nearest_line'].values
    
    # 归一化距离值
    distances_normalized = normalize_distances(distances, normalize_method)
    
    print(f"\n生成基于距离的多视图投影（{normalize_method}正规化）...")
    
    # ================= 1. 三维视图 (X-Y-Z) =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_norm, y_norm, z_norm, c=distances_normalized, s=2, alpha=0.6, cmap='viridis')
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(f'点到特征线的距离', fontsize=14)
    
    # 设置防拥挤边距
    # 【修改】：使用 mathtext 格式化坐标轴标签
    ax.set_xlabel(r'$\mathrm{X}$', labelpad=10)
    ax.set_ylabel(r'$\mathrm{Y}$', labelpad=10)
    ax.set_zlabel(r'$\mathrm{Z}$', labelpad=14)
    
    ax.tick_params(axis='x', pad=4)
    ax.tick_params(axis='y', pad=4)
    ax.tick_params(axis='z', pad=4)
    
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=-45)  # 保证Z轴在右侧
    
    # 【终极修复】：强制 ScalarFormatter 使用 mathtext 渲染负号
    formatter = plt.ScalarFormatter(useOffset=False)
    formatter.set_useMathText(True)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    
    ax.ticklabel_format(useOffset=False, style='plain', axis='both')
    ax.tick_params(axis='both', labelsize=14)
    
    # 【终极暴力修复】：强制所有刻度标签使用 Times New Roman
    fig.canvas.draw()
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    for label in tick_labels:
        label.set_fontname('Times New Roman')
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    
    save_path_png = os.path.join(img_dir, 'XYZ_3D_projection.png')
    save_path_svg = os.path.join(img_dir, 'XYZ_3D_projection.svg')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f"已保存: {save_path_png}")

    # ================= 2. 二维视图 (XY, XZ, YZ, XT, YT, ZT) =================
    # 【修改】：使用 mathtext 格式化坐标轴标签
    projections = [
        ('XY', x_norm, y_norm, r'$\mathrm{X}$', r'$\mathrm{Y}$'),
        ('XZ', x_norm, z_norm, r'$\mathrm{X}$', r'$\mathrm{Z}$'),
        ('YZ', y_norm, z_norm, r'$\mathrm{Y}$', r'$\mathrm{Z}$'),
        ('XT', t_norm, x_norm, r'$\mathrm{T}$', r'$\mathrm{X}$'),
        ('YT', t_norm, y_norm, r'$\mathrm{T}$', r'$\mathrm{Y}$'),
        ('ZT', t_norm, z_norm, r'$\mathrm{T}$', r'$\mathrm{Z}$')
    ]
    
    for title, px, py, xlabel, ylabel in projections:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(px, py, c=distances_normalized, s=2, alpha=0.6, cmap='viridis')
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, pad=0.05)
        cbar.set_label(f'点到特征线的距离', fontsize=14)
        
        ax.set_xlabel(f'{xlabel}', labelpad=10)
        ax.set_ylabel(f'{ylabel}', labelpad=10)
        
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
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        plt.tight_layout()
        save_path_png = os.path.join(img_dir, f'{title}_projection.png')
        save_path_svg = os.path.join(img_dir, f'{title}_projection.svg')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_svg, format='svg', dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {save_path_png}")

def normalize_distances(distances, normalize_method="percentile"):
    """
    归一化距离值
    
    Args:
        distances: 距离值数组
        normalize_method: 归一化方法
            - "percentile": 百分位数正规化（推荐，忽略极值）
            - "log": 对数正规化（适合跨度大的数据）
            - "minmax": 最小最大值正规化（原始方法）
    
    Returns:
        归一化后的距离值（0-1范围）
    """
    distances = np.array(distances)
    valid_mask = distances < np.inf
    distances_valid = distances[valid_mask]
    
    if len(distances_valid) == 0:
        print("⚠️  没有有效的距离值")
        return distances
    
    dist_min = np.min(distances_valid)
    dist_max = np.max(distances_valid)
    dist_mean = np.mean(distances_valid)
    dist_median = np.median(distances_valid)
    
    print(f"  距离统计:")
    print(f"    范围: [{dist_min:.2f}, {dist_max:.2f}]")
    print(f"    平均值: {dist_mean:.2f}, 中位数: {dist_median:.2f}")
    
    # 归一化距离
    distances_normalized = np.zeros_like(distances, dtype=float)
    
    if normalize_method == "percentile":
        # 百分位数正规化（推荐）
        percentile_low = np.percentile(distances_valid, 10)
        percentile_high = np.percentile(distances_valid, 90)
        print(f"    使用百分位数正规化: 2%-98% = [{percentile_low:.2f}, {percentile_high:.2f}]")
        
        if percentile_high > percentile_low:
            distances_normalized[valid_mask] = (distances[valid_mask] - percentile_low) / (percentile_high - percentile_low)
            distances_normalized[valid_mask] = np.clip(distances_normalized[valid_mask], 0, 1)
        else:
            distances_normalized[valid_mask] = 0.5
    
    elif normalize_method == "log":
        # 对数正规化（适合跨度大的数据）
        print(f"    使用对数正规化")
        log_distances = np.log1p(distances[valid_mask])  # log(1+x) 避免log(0)
        log_min = np.min(log_distances)
        log_max = np.max(log_distances)
        
        if log_max > log_min:
            distances_normalized[valid_mask] = (log_distances - log_min) / (log_max - log_min)
        else:
            distances_normalized[valid_mask] = 0.5
    
    else:  # minmax
        # 最小最大值正规化（可能被极值破坏）
        print(f"    使用最小最大值正规化")
        if dist_max > dist_min:
            distances_normalized[valid_mask] = (distances[valid_mask] - dist_min) / (dist_max - dist_min)
        else:
            distances_normalized[valid_mask] = 0.5
    
    distances_normalized[~valid_mask] = -0.1  # 无效点设为极端值
    
    return distances_normalized

def calculate_distance_to_line_segment(point, line_start, line_end):
    """计算点到线段的最短距离"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return np.linalg.norm(point - line_start)
    t = np.dot(point_vec, line_vec) / line_len
    t = max(0, min(1, t))
    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)

def main():
    # 读取原始点数据
    points_path = 'data/7303_xyzet_rotated.csv'
    points_df = pd.read_csv(points_path)
    points = points_df[['x', 'y', 'z']].values

    # 读取线参数数据
    lines_path = 'script_find_t_points/result/line_parameters.csv'
    lines_df = pd.read_csv(lines_path)

    print(f"原始微震点数: {len(points)}")
    print(f"时间轨迹线数: {len(lines_df)}")
    print("\n计算点到时间轨迹线的距离...")

    # 初始化存储最近距离的列表
    distances = []

    for i, point in enumerate(points):
        min_distance = float('inf')

        for _, line in lines_df.iterrows():
            line_start = np.array([line['x0'], line['y0'], line['z0']])
            t_min = line['t_min']
            t_max = line['t_max']
            line_end = line_start + (t_max - t_min) * np.array([0, 0, 1])

            distance = calculate_distance_to_line_segment(point, line_start, line_end)
            min_distance = min(min_distance, distance)

        distances.append(min_distance)
        
        if (i + 1) % 1000 == 0:
            print(f"  已处理: {i + 1:6d} / {len(points)}")

    distances = np.array(distances)
    print(f"  已处理: {len(points):6d} / {len(points)} ✓")

    # 将最近距离添加到原始点数据中
    points_df['distance_to_nearest_line'] = distances

    # 保存结果
    os.makedirs('script_calculate_distance/result', exist_ok=True)
    output_path = 'script_calculate_distance/result/t_point_distances.csv'
    points_df.to_csv(output_path, index=False)
    print(f"\n✓ 距离结果已保存: {output_path}")

    # 统计信息
    valid_distances = distances[distances < np.inf]
    print(f"\n距离统计:")
    print(f"  最小值: {np.min(valid_distances):.6f} m")
    print(f"  最大值: {np.max(valid_distances):.6f} m")
    print(f"  平均值: {np.mean(valid_distances):.6f} m")
    print(f"  中位数: {np.median(valid_distances):.6f} m")

    # ✅ 第一个3D窗口：离散彩虹色（与2D投影一致）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案1：离散彩虹色（与2D投影配色一致）")
    print("="*70)
    visualize_3d_point_cloud_with_distances(
        points, 
        distances,
        title="微震点到时间轨迹线的距离分布 - 离散彩虹色"
    )
    
    # ✅ 第二个3D窗口：连续色映射（更好地显示色差）
    print("\n" + "="*70)
    print("3D点云可视化 - 方案2：连续色映射（更清晰的色差）")
    print("="*70)
    visualize_3d_point_cloud_continuous(
        points, 
        distances,
        title="微震点到时间轨迹线的距离分布 - 连续Viridis色映射",
        cmap_name="viridis",
        normalize_method="percentile"  # ✅ 选择方法：percentile / log / minmax
    )
    
    # 原有的2D投影可视化
    print("\n生成2D投影可视化...")
    fig, point_colors = plot_xyzt_projections(
        points_df, 
        labels=points_df['distance_to_nearest_line'].values
    )
    plt.suptitle('微震点到时间轨迹线的距离分布 (2D投影 - 离散彩虹色)', fontsize=14, fontweight='bold')
    
    # 保存结果
    result_dir = 'script_calculate_distance/result'

    
    # 使用新的visualize_matplotlib_style函数生成多视图投影
    print("\n生成基于距离的多视图投影...")
    visualize_matplotlib_style(points_df, result_dir, normalize_method="percentile")
    print("\n🎉 距离计算及可视化完成！图表位于 script_calculate_distance/result/t_img/ 下")
    # plt.show() # 注释掉，防止阻塞

if __name__ == "__main__":
    main()
