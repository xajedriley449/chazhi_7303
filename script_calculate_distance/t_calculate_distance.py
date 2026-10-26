from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import plot_xyzt_projections, visualize_3d_point_cloud_with_distances, visualize_3d_point_cloud_continuous

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
    plt.show()

if __name__ == "__main__":
    main()