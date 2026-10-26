from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image  # 新增

from skimage.metrics import structural_similarity as ssim

def interpolate_images_linear(
    img1: np.ndarray,
    img2: np.ndarray,
    num_frames: int = 1
) -> List[np.ndarray]:
    """
    在两张图像之间线性插值生成中间帧。
    
    参数:
        img1: 第一张图像 [H, W]
        img2: 第二张图像 [H, W]
        num_frames: 插值帧数（不含首尾）
    
    返回:
        插值图像列表（不含img1和img2）
    """
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不匹配: {img1.shape} vs {img2.shape}")
    
    interpolated = []
    for i in range(1, num_frames + 1):
        # 线性插值权重
        alpha = i / (num_frames + 1)
        # 加权平均
        img_interp = (1 - alpha) * img1 + alpha * img2
        interpolated.append(img_interp)
    
    return interpolated


def generate_interpolated_sequence(
    image_dir: str,
    interpolation_factor: int = 1,
    pattern: str = "*.png"
) -> List[Tuple[np.ndarray, str]]:
    """
    加载图像序列并在相邻帧间插值。
    
    参数:
        image_dir: 原始图像目录
        interpolation_factor: 插值密度因子
            - 1: 不插值
            - 2: 每两帧间插1帧
            - 4: 每两帧间插3帧
            - n: 每两帧间插(n-1)帧
        pattern: 文件匹配模式
    
    返回:
        [(图像数组, 标签), ...] 列表
    """
    import glob
    
    # 加载原始图像
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if len(image_paths) < 2:
        raise ValueError(f"图像数量不足（至少需要2张），当前: {len(image_paths)}")
    
    original_images = []
    for path in image_paths:
        img = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
        name = os.path.splitext(os.path.basename(path))[0]
        original_images.append((img, name))
    
    # 不插值，直接返回
    if interpolation_factor == 1:
        return original_images
    
    # 插值
    num_interp_frames = interpolation_factor - 1
    result = []
    
    for i in range(len(original_images) - 1):
        img1, name1 = original_images[i]
        img2, name2 = original_images[i + 1]
        
        # 添加第一张原图
        result.append((img1, f"{name1}_orig"))
        
        # 生成插值帧
        interp_imgs = interpolate_images_linear(img1, img2, num_interp_frames)
        for j, img_interp in enumerate(interp_imgs):
            label = f"{name1}_to_{name2}_interp{j+1:02d}"
            result.append((img_interp, label))
    
    # 添加最后一张原图
    result.append((original_images[-1][0], f"{original_images[-1][1]}_orig"))
    
    print(f"插值因子={interpolation_factor}, 原始帧={len(original_images)}, 总帧数={len(result)}")
    return result


def process_with_image_interpolation(
    df: pd.DataFrame,
    interpolation_factor: int,
    base_output_dir: str = "output"
) -> None:
    """
    使用图像插值处理数据并生成SSIM分析。
    
    参数:
        df: 归一化后的数据框（已包含_datetime列）
        interpolation_factor: 插值密度因子
        base_output_dir: 输出根目录
    """
    # 创建专用输出目录
    output_dir = os.path.join(base_output_dir, f"ssim_{interpolation_factor}")
    os.makedirs(output_dir, exist_ok=True)
    
    daily_dir = os.path.join(output_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"处理插值因子: {interpolation_factor} (图像插值模式)")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    # ==================== 第一步：生成原始每日图像 ====================
    temp_dir = os.path.join(output_dir, "temp_originals")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 按日期分组
    unique_dates = sorted(df["_date"].unique())
    print(f"共有 {len(unique_dates)} 个日期")
    
    H, W = 256, 256
    original_image_paths = {"tx": [], "ty": [], "tz": []}
    
    for date in unique_dates:
        date_df = df[df["_date"] == date].copy()
        if date_df.empty:
            continue
        
        # 归一化t列
        target_cols = [c for c in ["t"] if c in date_df.columns]
        date_norm, _ = normalize_df_minmax(date_df, target_cols)
        
        # 绘制
        fig, img_tx, img_ty, img_tz = plot_tx_ty_tz_rainbow_max(
            date_norm,
            grid=256,
            radius_scale=0.2,
            use_cuda=True,
            max_points=None,
            show=False,
            d=date,
        )
        plt.close(fig)
        
        # 保存原始图像
        for proj, img in [("tx", img_tx), ("ty", img_ty), ("tz", img_tz)]:
            save_path = os.path.join(temp_dir, proj, f"{date}.png")
            save_gray_img(img, save_path)
            original_image_paths[proj].append(save_path)
    
    print(f"已生成 {len(unique_dates)} 张原始图像")
    
    # ==================== 第二步：对每个投影方向进行插值 ====================
    for proj in ["tx", "ty", "tz"]:
        print(f"\n处理 {proj.upper()} 投影的插值...")
        proj_temp_dir = os.path.join(temp_dir, proj)
        proj_output_dir = os.path.join(daily_dir, proj)
        os.makedirs(proj_output_dir, exist_ok=True)
        
        # 生成插值序列
        interp_sequence = generate_interpolated_sequence(
            proj_temp_dir,
            interpolation_factor=interpolation_factor,
            pattern="*.png"
        )
        
        # 保存插值后的图像
        for img, label in interp_sequence:
            save_path = os.path.join(proj_output_dir, f"{label}.png")
            save_gray_img(img, save_path)
        
        print(f"  {proj.upper()} 插值完成，共 {len(interp_sequence)} 帧")
    
    # ==================== 第三步：计算SSIM ====================
    print("\n开始计算相邻帧SSIM...")
    
    projections = [
        ("tx", os.path.join(daily_dir, "tx"), "T-X Projection"),
        ("ty", os.path.join(daily_dir, "ty"), "T-Y Projection"),
        ("tz", os.path.join(daily_dir, "tz"), "T-Z Projection"),
    ]
    
    for proj_name, proj_dir, proj_title in projections:
        try:
            # 计算相邻帧SSIM
            pair_labels, ssim_scores = compute_adjacent_ssim(proj_dir, "*.png")
            
            print(f"\n{proj_title} SSIM统计:")
            print(f"  平均值: {np.mean(ssim_scores):.4f}")
            print(f"  中位数: {np.median(ssim_scores):.4f}")
            print(f"  标准差: {np.std(ssim_scores):.4f}")
            print(f"  最小值: {np.min(ssim_scores):.4f}")
            print(f"  最大值: {np.max(ssim_scores):.4f}")
            
            # 保存统计数据
            csv_path = os.path.join(output_dir, f"ssim_{proj_name}_statistics.csv")
            save_ssim_statistics(pair_labels, ssim_scores, csv_path)
            
            # 绘制SSIM曲线
            curve_path = os.path.join(output_dir, f"ssim_{proj_name}_curve.png")
            plot_ssim_curve(
                pair_labels,
                ssim_scores,
                save_path=curve_path,
                title=f"Adjacent Frames SSIM - {proj_title} (ImageInterp={interpolation_factor})",
                figsize=(16, 6),
                show=False,
            )
            
        except Exception as e:
            print(f"计算 {proj_name} SSIM时出错: {e}")
            continue
    
    # 绘制综合对比图
    try:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        for idx, (proj_name, proj_dir, proj_title) in enumerate(projections):
            pair_labels, ssim_scores = compute_adjacent_ssim(proj_dir, "*.png")
            x = np.arange(len(ssim_scores))
            
            ax = axes[idx]
            ax.plot(x, ssim_scores, marker='o', linewidth=2, markersize=5, label=proj_title)
            ax.axhline(y=np.mean(ssim_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(ssim_scores):.4f}')
            ax.set_xlabel('Frame Pair Index', fontsize=11)
            ax.set_ylabel('SSIM Score', fontsize=11)
            ax.set_title(f'{proj_title} - Adjacent Frames SSIM (ImageInterp={interpolation_factor})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='lower left', fontsize=10)
            ax.set_ylim([0, 1])
            
            step = max(1, len(x) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(x[::step], fontsize=9)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, "ssim_all_projections_comparison.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"\n综合SSIM对比图已保存: {combined_path}")
        plt.close()
        
    except Exception as e:
        print(f"绘制综合对比图时出错: {e}")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n插值因子 {interpolation_factor} 处理完成（图像插值模式）！")

def process_with_interpolation_factor(
    df: pd.DataFrame,
    interpolation_factor: int,
    base_output_dir: str = "output"
) -> None:
    """
    使用指定的插值因子处理数据并生成SSIM分析。
    
    参数:
        df: 归一化后的数据框（已包含_datetime列）
        interpolation_factor: 插值密度因子
        base_output_dir: 输出根目录
    """
    # 创建专用输出目录
    output_dir = os.path.join(base_output_dir, f"ssim_{interpolation_factor}")
    os.makedirs(output_dir, exist_ok=True)
    
    daily_dir = os.path.join(output_dir, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"处理插值因子: {interpolation_factor}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    # 导入滑动窗口函数
    from utils import generate_sliding_windows
    
    # 生成滑动窗口
    windows = generate_sliding_windows(
        df,
        date_col="_date",
        datetime_col="_datetime",
        window_hours=24.0,
        interpolation_factor=interpolation_factor,
    )
    
    H, W = 256, 256
    # 横向长图：宽度按窗口数量
    img_tx_concat = np.zeros((H, W * len(windows)), dtype=np.float32)
    img_ty_concat = np.zeros((H, W * len(windows)), dtype=np.float32)
    img_tz_concat = np.zeros((H, W * len(windows)), dtype=np.float32)
    image_paths = []
    
    for i, (window_df, label) in enumerate(windows):
        if window_df.empty:
            continue
        
        # 对窗口数据归一化t列（每个窗口独立归一化）
        target_cols = [c for c in ["t"] if c in window_df.columns]
        window_norm, _ = normalize_df_minmax(window_df, target_cols)
        
        # 绘制并保存（不show）
        fig, img_tx, img_ty, img_tz = plot_tx_ty_tz_rainbow_max(
            window_norm,
            grid=256,
            radius_scale=0.2,
            use_cuda=True,
            max_points=None,
            show=False,
            d=label,
        )
        
        out_png = os.path.join(daily_dir, "tx_ty_tz", f"{label}.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        image_paths.append(out_png)
        
        # 释放内存
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # 保存纯图像
        save_gray_img(img_tx, os.path.join(daily_dir, "tx", f"{label}.png"))
        save_gray_img(img_ty, os.path.join(daily_dir, "ty", f"{label}.png"))
        save_gray_img(img_tz, os.path.join(daily_dir, "tz", f"{label}.png"))
        
        # 拼接到横向长图
        c0, c1 = i * W, (i + 1) * W
        img_tx_concat[:, c0:c1] = img_tx
        img_ty_concat[:, c0:c1] = img_ty
        img_tz_concat[:, c0:c1] = img_tz
        
        if (i + 1) % 10 == 0 or (i + 1) == len(windows):
            print(f"  完成窗口 {i+1}/{len(windows)}")
    
    # 保存横向长图
    save_gray_img(img_tx_concat, os.path.join(daily_dir, "img_tx_concat.png"))
    save_gray_img(img_ty_concat, os.path.join(daily_dir, "img_ty_concat.png"))
    save_gray_img(img_tz_concat, os.path.join(daily_dir, "img_tz_concat.png"))
    
    print(f"生成 {len(image_paths)} 张图像")
    
    # 纵向拼接可视化（可选，不显示）
    if image_paths:
        concat_path = os.path.join(daily_dir, "daily_concat.png")
        concat_images_vertical(sorted(image_paths), out_path=concat_path)
        print(f"已生成长图: {concat_path}")
    
    # ==================== 计算并绘制相邻帧SSIM ====================
    print("\n开始计算相邻帧SSIM...")
    
    # 导入SSIM计算和绘图函数
    from utils import compute_adjacent_ssim, plot_ssim_curve, save_ssim_statistics
    
    # 分别对三个投影方向计算SSIM
    projections = [
        ("tx", os.path.join(daily_dir, "tx"), "T-X Projection"),
        ("ty", os.path.join(daily_dir, "ty"), "T-Y Projection"),
        ("tz", os.path.join(daily_dir, "tz"), "T-Z Projection"),
    ]
    
    for proj_name, proj_dir, proj_title in projections:
        if not os.path.exists(proj_dir):
            print(f"跳过 {proj_name}：目录不存在")
            continue
        
        try:
            # 计算相邻帧SSIM
            pair_labels, ssim_scores = compute_adjacent_ssim(proj_dir, "*.png")
            
            print(f"\n{proj_title} SSIM统计:")
            print(f"  平均值: {np.mean(ssim_scores):.4f}")
            print(f"  中位数: {np.median(ssim_scores):.4f}")
            print(f"  标准差: {np.std(ssim_scores):.4f}")
            print(f"  最小值: {np.min(ssim_scores):.4f}")
            print(f"  最大值: {np.max(ssim_scores):.4f}")
            
            # 保存统计数据
            csv_path = os.path.join(output_dir, f"ssim_{proj_name}_statistics.csv")
            save_ssim_statistics(pair_labels, ssim_scores, csv_path)
            
            # 绘制并保存SSIM曲线
            curve_path = os.path.join(output_dir, f"ssim_{proj_name}_curve.png")
            plot_ssim_curve(
                pair_labels,
                ssim_scores,
                save_path=curve_path,
                title=f"Adjacent Frames SSIM - {proj_title} (Factor={interpolation_factor})",
                figsize=(16, 6),
                show=False,
            )
            
        except Exception as e:
            print(f"计算 {proj_name} SSIM时出错: {e}")
            continue
    
    # 绘制综合对比图（三个投影的SSIM在一张图上）
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        for idx, (proj_name, proj_dir, proj_title) in enumerate(projections):
            if not os.path.exists(proj_dir):
                continue
            
            pair_labels, ssim_scores = compute_adjacent_ssim(proj_dir, "*.png")
            x = np.arange(len(ssim_scores))
            
            ax = axes[idx]
            ax.plot(x, ssim_scores, marker='o', linewidth=2, markersize=5, label=proj_title)
            ax.axhline(y=np.mean(ssim_scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(ssim_scores):.4f}')
            ax.set_xlabel('Frame Pair Index', fontsize=11)
            ax.set_ylabel('SSIM Score', fontsize=11)
            ax.set_title(f'{proj_title} - Adjacent Frames SSIM (Factor={interpolation_factor})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='lower left', fontsize=10)
            ax.set_ylim([0, 1])
            
            # 设置x轴刻度
            step = max(1, len(x) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(x[::step], fontsize=9)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, "ssim_all_projections_comparison.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"\n综合SSIM对比图已保存: {combined_path}")
        plt.close()
        
    except Exception as e:
        print(f"绘制综合对比图时出错: {e}")
    
    print(f"\n插值因子 {interpolation_factor} 处理完成！")

def generate_multi_factor_comparison(factors: List[int], base_dir: str) -> None:
    """生成多个插值因子的SSIM对比图"""
    import matplotlib.pyplot as plt
    from utils import compute_adjacent_ssim
    
    projections = ["tx", "ty", "tz"]
    proj_titles = {"tx": "T-X Projection", "ty": "T-Y Projection", "tz": "T-Z Projection"}
    
    for proj in projections:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for factor in factors:
            proj_dir = os.path.join(base_dir, f"ssim_{factor}", "daily", proj)
            if not os.path.exists(proj_dir):
                continue
            
            try:
                _, ssim_scores = compute_adjacent_ssim(proj_dir, "*.png")
                x = np.arange(len(ssim_scores))
                
                ax.plot(x, ssim_scores, marker='o', linewidth=2, markersize=4, 
                       label=f'Factor={factor} (Mean={np.mean(ssim_scores):.4f})', alpha=0.7)
            except Exception:
                continue
        
        ax.set_xlabel('Frame Pair Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{proj_titles[proj]} - Multi-Factor SSIM Comparison', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = os.path.join(base_dir, f"ssim_{proj}_multi_factor_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

def compute_adjacent_ssim(image_dir: str, pattern: str = "*.png") -> Tuple[List[str], List[float]]:
    """
    计算目录中相邻图像的SSIM值。
    
    参数:
        image_dir: 图像目录路径
        pattern: 文件匹配模式
    
    返回:
        (图像对标签列表, SSIM值列表)
    """
    import glob
    
    # 获取所有图像路径并排序
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    
    if len(image_paths) < 2:
        raise ValueError(f"图像数量不足（需要至少2张），当前: {len(image_paths)}")
    
    ssim_scores = []
    pair_labels = []
    
    for i in range(len(image_paths) - 1):
        # 加载相邻两张图像
        img1 = np.array(Image.open(image_paths[i]).convert('L'), dtype=np.float32) / 255.0
        img2 = np.array(Image.open(image_paths[i + 1]).convert('L'), dtype=np.float32) / 255.0
        
        # 计算SSIM
        score = ssim(img1, img2, data_range=1.0)
        ssim_scores.append(score)
        
        # 生成标签（相邻帧对）
        name1 = os.path.splitext(os.path.basename(image_paths[i]))[0]
        name2 = os.path.splitext(os.path.basename(image_paths[i + 1]))[0]
        pair_labels.append(f"{name1}\n→\n{name2}")
    
    return pair_labels, ssim_scores


def plot_ssim_curve(
    pair_labels: List[str],
    ssim_scores: List[float],
    save_path: str,
    title: str = "Adjacent Frames SSIM",
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True
) -> None:
    """
    绘制SSIM折线图。
    
    参数:
        pair_labels: 图像对标签列表
        ssim_scores: SSIM值列表
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        show: 是否显示图表
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(ssim_scores))
    
    # 绘制折线和散点
    ax.plot(x, ssim_scores, marker='o', linewidth=2, markersize=6, 
            color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', 
            markeredgewidth=1.5, label='SSIM')
    
    # 填充区域
    ax.fill_between(x, ssim_scores, alpha=0.2, color='#2E86AB')
    
    # 添加平均线
    mean_ssim = np.mean(ssim_scores)
    ax.axhline(y=mean_ssim, color='#F18F01', linestyle='--', linewidth=2, 
               label=f'Mean SSIM: {mean_ssim:.4f}')
    
    # 标注最小值和最大值
    min_idx = np.argmin(ssim_scores)
    max_idx = np.argmax(ssim_scores)
    ax.scatter([min_idx], [ssim_scores[min_idx]], color='red', s=100, 
               zorder=5, marker='v', label=f'Min: {ssim_scores[min_idx]:.4f}')
    ax.scatter([max_idx], [ssim_scores[max_idx]], color='green', s=100, 
               zorder=5, marker='^', label=f'Max: {ssim_scores[max_idx]:.4f}')
    
    # 设置标签和标题
    ax.set_xlabel('Frame Pair Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置x轴刻度（每隔几个显示，避免过密）
    step = max(1, len(x) // 15)  # 最多显示15个刻度
    ax.set_xticks(x[::step])
    ax.set_xticklabels([pair_labels[i] for i in range(0, len(pair_labels), step)], 
                       rotation=45, ha='right', fontsize=8)
    
    # 设置y轴范围
    ax.set_ylim([max(0, min(ssim_scores) - 0.05), min(1, max(ssim_scores) + 0.05)])
    
    # 网格和图例
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # 保存
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"SSIM曲线已保存: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def save_ssim_statistics(
    pair_labels: List[str],
    ssim_scores: List[float],
    save_path: str
) -> None:
    """
    保存SSIM统计数据到CSV文件。
    
    参数:
        pair_labels: 图像对标签列表
        ssim_scores: SSIM值列表
        save_path: 保存路径
    """
    df = pd.DataFrame({
        'Frame_Pair': pair_labels,
        'SSIM': ssim_scores
    })
    
    # 添加统计信息
    stats = pd.DataFrame({
        'Frame_Pair': ['Mean', 'Median', 'Std', 'Min', 'Max'],
        'SSIM': [
            np.mean(ssim_scores),
            np.median(ssim_scores),
            np.std(ssim_scores),
            np.min(ssim_scores),
            np.max(ssim_scores)
        ]
    })
    
    # 合并数据
    result = pd.concat([df, pd.DataFrame([['---', '---']]), stats], ignore_index=True)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    result.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"SSIM统计数据已保存: {save_path}")

def generate_sliding_windows(
    df: pd.DataFrame,
    date_col: str = "_date",
    datetime_col: str = "_datetime",
    window_hours: float = 24.0,
    interpolation_factor: int = 1,  # 新增参数
) -> List[Tuple[pd.DataFrame, str]]:
    """
    生成滑动窗口数据帧用于插值。
    
    参数:
        df: 包含时间信息的数据框
        date_col: 日期列名
        datetime_col: datetime列名（需事先添加）
        window_hours: 滑动窗口大小（小时），默认24
        interpolation_factor: 插值密度因子
            - 1: 不插值（slide=24h，每日1帧）
            - 2: 插值1帧（slide=12h，每日2帧）
            - 4: 插值3帧（slide=6h，每日4帧）
            - 8: 插值7帧（slide=3h，每日8帧）
            - n: 插值(n-1)帧（slide=24/n h，每日n帧）
    
    返回:
        [(窗口数据帧, 标签字符串), ...]
    """
    if datetime_col not in df.columns:
        raise ValueError(f"缺少列: {datetime_col}，请先生成完整datetime列")
    
    if interpolation_factor < 1:
        raise ValueError(f"interpolation_factor必须>=1，当前值: {interpolation_factor}")
    
    # 计算滑动步长
    slide_hours = window_hours / interpolation_factor
    
    df = df.sort_values(datetime_col).reset_index(drop=True)
    start_time = df[datetime_col].min()
    end_time = df[datetime_col].max()
    
    windows = []
    current_time = start_time
    
    from datetime import timedelta
    window_delta = timedelta(hours=window_hours)
    slide_delta = timedelta(hours=slide_hours)
    
    frame_idx = 0
    while current_time <= end_time:
        window_end = current_time + window_delta
        
        # 提取当前窗口内的数据
        mask = (df[datetime_col] >= current_time) & (df[datetime_col] < window_end)
        window_data = df[mask].copy()
        
        if len(window_data) > 0:
            # 生成标签：窗口中心时间
            center_time = current_time + window_delta / 2
            label = f"{center_time.strftime('%Y-%m-%d_%H%M')}_frame{frame_idx:03d}"
            windows.append((window_data, label))
            frame_idx += 1
        
        current_time += slide_delta
    
    print(f"插值因子={interpolation_factor}, 滑动步长={slide_hours:.1f}h, 生成帧数={len(windows)}")
    return windows

def save_gray_img(img01, save_path, origin: str = "upper"):
    """
    将 2D 数组保存成纯灰度图(无坐标轴/边框/网格)。
    - 支持 numpy/cupy 数组
    - 假定输入为 [0,1] 浮点或 uint8，保存为 8-bit 灰度 PNG
    - origin: "upper"(默认) 或 "lower"。若为"lower"，保存前做上下翻转以匹配 Matplotlib origin="lower" 的视觉效果。
    """
    if hasattr(img01, "get"):  # cupy -> numpy
        img01 = img01.get()
    arr = img01
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if origin.lower() == "lower":
        arr = np.flipud(arr)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    Image.fromarray(arr, mode="L").save(save_path)

def load_csv(path: str, **read_csv_kwargs) -> pd.DataFrame:
	"""读取CSV为DataFrame。"""
	defaults = dict(engine="c", encoding="utf-8")
	defaults.update(read_csv_kwargs or {})
	return pd.read_csv(path, **defaults)


def save_csv(df: pd.DataFrame, path: str, **to_csv_kwargs) -> None:
	"""保存DataFrame为CSV。"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	defaults = dict(index=False, encoding="utf-8")
	defaults.update(to_csv_kwargs or {})
	df.to_csv(path, **defaults)


def normalize_df_minmax(
	df: pd.DataFrame,
	columns: Iterable[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
	"""
	对指定列进行Min-Max归一化到[0,1]区间。
	返回：归一化后的副本，以及每列的min/max字典。
	"""
	out = df.copy()
	stats: Dict[str, Dict[str, float]] = {}
	for col in columns:
		if col not in out.columns:
			continue
		col_values = pd.to_numeric(out[col], errors="coerce")
		cmin = float(np.nanmin(col_values))
		cmax = float(np.nanmax(col_values))
		den = (cmax - cmin)
		# 避免除零
		if den == 0 or np.isclose(den, 0.0):
			norm = np.zeros_like(col_values, dtype=float)
		else:
			norm = (col_values - cmin) / den
		out[col] = norm
		stats[col] = {"min": cmin, "max": cmax}
	return out, stats


def plot_pair(
	df: pd.DataFrame,
	x_col: str,
	y_col: str,
	title: Optional[str] = None,
	ax: Optional[plt.Axes] = None,
	alpha: float = 0.6,
	s: float = 6.0,
	color: str = "#1f77b4",
) -> plt.Axes:
	"""绘制单个二维散点图。"""
	ax = ax or plt.gca()
	ax.scatter(df[x_col], df[y_col], s=s, alpha=alpha, c=color)
	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	if title:
		ax.set_title(title, fontsize=10)
	ax.grid(True, linestyle="--", alpha=0.3)
	return ax


def plot_pairs(
	df: pd.DataFrame,
	pairs: Iterable[Tuple[str, str]],
	ncols: int = 3,
	figsize: Tuple[int, int] = (12, 6),
	suptitle: Optional[str] = None,
	tight_layout: bool = True,
	show: bool = True,
) -> plt.Figure:
	"""
	成组绘制多个二维散点图。
	pairs: 例如 [("x","y"), ("x","z"), ("y","z"), ("x","t"), ("y","t"), ("z","t")]
	"""
	pairs = list(pairs)
	n = len(pairs)
	ncols = max(1, ncols)
	nrows = (n + ncols - 1) // ncols

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
	if not isinstance(axes, np.ndarray):
		axes = np.array([axes])
	axes = axes.reshape(nrows, ncols)

	for idx, (x_col, y_col) in enumerate(pairs):
		r, c = divmod(idx, ncols)
		ax = axes[r, c]
		title = f"{x_col} vs {y_col}"
		plot_pair(df, x_col, y_col, title=title, ax=ax)

	# 隐藏多余子图
	for k in range(n, nrows * ncols):
		r, c = divmod(k, ncols)
		axes[r, c].axis("off")

	if suptitle:
		fig.suptitle(suptitle, fontsize=12)
	if tight_layout:
		plt.tight_layout(rect=[0, 0.03, 1, 0.95] if suptitle else None)
	if show:
		plt.show()
	return fig


def plot_final_projections(df):

    # 创建6个子图
    fig = plt.figure(figsize=(18, 12))
    
    # XY平面
    ax1 = fig.add_subplot(231)
    ax1.scatter(df['x'], df['y'], s=1)
    ax1.set_title('XY Projection')
    
    # XZ平面
    ax2 = fig.add_subplot(232)
    ax2.scatter(df['x'], df['z'], s=1)
    ax2.set_title('XZ Projection')
    
    # YZ平面
    ax3 = fig.add_subplot(233)
    ax3.scatter(df['y'], df['z'], s=1)
    ax3.set_title('YZ Projection')
    
    # XT平面
    ax4 = fig.add_subplot(234)
    ax4.scatter(df['t'], df['x'], s=1)
    ax4.set_title('TX Projection')
    
    # YT平面
    ax5 = fig.add_subplot(235)
    ax5.scatter(df['t'], df['y'], s=1)
    ax5.set_title('TY Projection')
    
    # ZT平面
    ax6 = fig.add_subplot(236)
    ax6.scatter(df['t'], df['z'], s=1)
    ax6.set_title('TZ Projection')
    
    plt.tight_layout()
    return fig

def plot_xyzt_projections(df, labels):
    
    # 确保 labels 是 NumPy 数组
    labels = np.array(labels)

    # 生成颜色映射
    unique_clusters = sorted(np.unique(labels))  # 使用 numpy.unique 获取唯一值
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

    # 创建颜色字典，噪声点为灰色
    color_dict = {-1: '#808080'}  # 灰色
    for i, cluster in enumerate(unique_clusters):
        color_dict[cluster] = colors[i]

    # 创建颜色列表
    point_colors = [color_dict[label] for label in labels]  # 使用列表推导替代 map
    
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
    return fig, point_colors

def _rainbow_purple_to_red_cmap(name: str = "purple_to_red_rainbow") -> LinearSegmentedColormap:
	"""
	构建紫->蓝->青->绿->黄->橙->红的彩虹色映射。
	0=紫, 1=红（全局统一范围）。
	"""
	colors = [
		"#800080",  # purple
		"#0000FF",  # blue
		"#00FFFF",  # cyan
		"#00FF00",  # green
		"#FFFF00",  # yellow
		"#FFA500",  # orange
		"#FF0000",  # red
	]
	return LinearSegmentedColormap.from_list(name, colors, N=256)


def plot_tx_ty_tz_rainbow_max(
	df,
	grid: int = 512,
	radius_scale: float = 0.25,
	use_cuda: bool = True,
	max_points: int | None = 5000,
	show: bool = True,  # 新增：是否show
	d: Optional[str] = None,  # 新增：日期标签
):
	"""
	按每个微震点构建彩色圆（二维圆锥截面），并以逐像素最大值(max)覆盖合成，绘制 t-x / t-y / t-z。
	- 颜色: 彩虹(紫->红)，全局0..1范围；0=紫，1=红。
	- 半径: r_pix = radius_scale * e * grid（e需已归一化到[0,1]）。
	- 像素值: val = e * max(0, 1 - dist/r)，多点合成取max。
	- 性能: 优先使用CUDA(CuPy)，否则回退CPU；可通过max_points抽样以控制耗时。
	参数:
	  grid: 栅格大小（每轴像素数）
	  radius_scale: 半径缩放系数（与grid线性相关）
	  use_cuda: 优先使用GPU
	  max_points: 最多绘制的点数（None=全部，注意可能很慢）
	  show: 是否在生成后调用 plt.show()
	"""
	# 列检查
	for col in ("t", "x", "y", "z", "e"):
		if col not in df.columns:
			raise ValueError(f"缺少列: {col}")

	# 选择后端
	xp = np
	using_cuda = False
	if use_cuda:
		try:
			import cupy as cp  # type: ignore
			xp = cp
			using_cuda = True
		except Exception:
			using_cuda = False

	# 数据准备与抽样
	data = df[["t", "x", "y", "z", "e"]].to_numpy(copy=False)
	n = len(data)
	if max_points is not None and n > max_points:
		# 固定随机种子保证可重复
		sampled = df.sample(n=max_points, random_state=42)[["t", "x", "y", "z", "e"]].to_numpy(copy=False)
		data = sampled
		n = len(data)

	t = xp.asarray(data[:, 4], dtype=xp.float32)  # 警惕列顺序，这里修正：data[:,4]是e？我们重排明确
	# 纠正：data列顺序 ["t","x","y","z","e"]，分别索引0/1/2/3/4
	t = xp.asarray(data[:, 0], dtype=xp.float32)
	x = xp.asarray(data[:, 1], dtype=xp.float32)
	y = xp.asarray(data[:, 2], dtype=xp.float32)
	z = xp.asarray(data[:, 3], dtype=xp.float32)
	e = xp.asarray(data[:, 4], dtype=xp.float32)

	# 归一化域固定[0,1]=>像素索引
	def to_idx(a):
		return xp.clip((a * (grid - 1)).astype(xp.int32), 0, grid - 1)

	# 渲染函数：max覆盖
	def render_max(u, v):
		img = xp.zeros((grid, grid), dtype=xp.float32)
		# 每点半径（像素）
		r_pix = xp.maximum(1, (radius_scale * e * grid).astype(xp.int32))
		iu = to_idx(u)  # 列(x轴)
		iv = to_idx(v)  # 行(y轴)

		# 分组复用核
		r_unique = xp.unique(r_pix)
		for r in r_unique.tolist():
			r = int(r)
			# 该半径的点索引
			mask = (r_pix == r)
			if not bool(mask.sum()):
				continue

			# 预计算核 k，中心(0,0)=1, 边缘=0
			ax = xp.arange(-r, r + 1, dtype=xp.float32)
			yy, xx = xp.meshgrid(ax, ax, indexing="ij")
			dist = xp.sqrt(xx * xx + yy * yy)
			k = xp.maximum(0.0, 1.0 - dist / float(r))

			# 逐点max覆盖
			idxs = xp.where(mask)[0].tolist()
			for idx in idxs:
				cx = int(iu[idx])
				cy = int(iv[idx])
				val = float(e[idx])
				if val <= 0.0:
					continue

				# 贴图边界
				y0 = max(0, cy - r); y1 = min(grid, cy + r + 1)
				x0 = max(0, cx - r); x1 = min(grid, cx + r + 1)

				# 核裁剪
				ky0 = r - (cy - y0)
				kx0 = r - (cx - x0)
				ky1 = ky0 + (y1 - y0)
				kx1 = kx0 + (x1 - x0)

				patch = val * k[ky0:ky1, kx0:kx1]
				# 最大值覆盖
				img[y0:y1, x0:x1] = xp.maximum(img[y0:y1, x0:x1], patch)

		# 返回CPU数组
		return img.get() if using_cuda else img

	# 分别渲染三个平面
	img_tx = render_max(t, x)
	img_ty = render_max(t, y)
	img_tz = render_max(t, z)

	# 彩虹色映射
	cmap = _rainbow_purple_to_red_cmap()
	#cmap = 'viridis'

	# 显示
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	for ax, img, ylabel, title in [
		(axes[0], img_tx, "x", "t-x"),
		(axes[1], img_ty, "y", "t-y"),
		(axes[2], img_tz, "z", "t-z"),
	]:
		ax.imshow(img, origin="upper", extent=[0, 1, 0, 1], cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
		ax.set_xlabel("t")
		ax.set_ylabel(ylabel)
		ax.set_title(title, fontsize=10)
		ax.grid(True, linestyle="--", alpha=0.3)
	plt.tight_layout()
	if show:  # 修改：可控显示
		plt.show()
	return fig, img_tx, img_ty, img_tz

def concat_images_vertical(image_paths: List[str], out_path: Optional[str] = None, bg_color=(255, 255, 255)) -> Image.Image:
	"""将多张图片按给定顺序纵向拼接，保存到 out_path（可选），返回拼接后的PIL Image。"""
	images: List[Image.Image] = []
	for p in image_paths:
		try:
			im = Image.open(p).convert("RGB")
			images.append(im)
		except Exception:
			continue
	if not images:
		raise ValueError("没有可拼接的图像。")

	max_w = max(im.width for im in images)
	resized: List[Image.Image] = []
	total_h = 0
	for im in images:
		if im.width != max_w:
			new_h = int(im.height * (max_w / im.width))
			im = im.resize((max_w, new_h), Image.BILINEAR)
		resized.append(im)
		total_h += im.height

	canvas = Image.new("RGB", (max_w, total_h), bg_color)
	y = 0
	for im in resized:
		canvas.paste(im, (0, y))
		y += im.height

	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		canvas.save(out_path)
	return canvas

# 在文件末尾添加

# 更优化的版本（按百分位分组）

def visualize_3d_point_cloud_with_distances(xyz, distances, title: str = "3D Point Cloud Distance Distribution", n_colors: int = 20):
    """
    可视化3D点云，颜色方案与plot_xyzt_projections一致（离散彩虹色）
    
    Args:
        xyz: 点的xyz坐标 [[x1,y1,z1], ...]
        distances: 每个点的距离值 [d1, d2, ...]
        title: 窗口标题
        n_colors: 离散颜色数量（默认20种，避免过多）
    """
    import open3d as o3d
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    distances = np.array(distances)
    valid_mask = distances < np.inf
    distances_valid = distances[valid_mask]
    
    if len(distances_valid) == 0:
        print("⚠️  没有有效的距离值")
        return
    
    dist_min = np.min(distances_valid)
    dist_max = np.max(distances_valid)
    
    # ✅ 将距离值分组到离散的颜色级别
    # 使用 np.linspace 创建 n_colors 个等间距的距离区间
    distance_bins = np.linspace(dist_min, dist_max, n_colors + 1)
    distance_labels = np.digitize(distances, distance_bins) - 1  # 返回分组索引
    distance_labels = np.clip(distance_labels, 0, n_colors - 1)  # 确保在范围内
    
    # 生成颜色映射（与 plot_xyzt_projections 逻辑完全一致）
    colors_map = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    
    # 创建颜色字典
    color_dict = {}
    for i in range(n_colors):
        rgba = colors_map[i]
        color_dict[i] = rgba[:3]  # 取前3个值 (R, G, B)
    
    # 为无效点设置灰色
    color_dict[-1] = np.array([0.5, 0.5, 0.5])
    
    # 创建颜色列表
    point_colors = []
    for dist, label in zip(distances, distance_labels):
        if np.isinf(dist):
            point_colors.append([0.5, 0.5, 0.5])  # 灰色
        else:
            point_colors.append(color_dict[int(label)])
    
    point_colors = np.array(point_colors)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 显示点云
    print("生成3D点云可视化...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{title}\n最小值={dist_min:.2f}m, 最大值={dist_max:.2f}m, 平均值={np.mean(distances_valid):.2f}m\n(离散彩虹色×{n_colors}，与2D投影配色一致)",
        width=1200,
        height=800
    )
    
# 在 visualize_3d_point_cloud_with_distances 函数之后添加

# 替换原有的 visualize_3# 备选方案：对数正规化

def visualize_3d_point_cloud_continuous(xyz, distances, title: str = "3D Point Cloud Distance Distribution (Continuous)", 
                                        cmap_name: str = "viridis", normalize_method: str = "percentile"):
    """
    可视化3D点云，支持多种正规化方法
    
    Args:
        xyz: 点的xyz坐标
        distances: 每个点的距离值
        title: 窗口标题
        cmap_name: colormap名称
        normalize_method: 正规化方法
            - "percentile": 百分位数正规化（推荐，忽略极值）
            - "log": 对数正规化（适合跨度大的数据）
            - "minmax": 最小最大值正规化（原始方法）
    """
    import open3d as o3d
    import matplotlib.cm as cm
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    distances = np.array(distances)
    valid_mask = distances < np.inf
    distances_valid = distances[valid_mask]
    
    if len(distances_valid) == 0:
        print("⚠️  没有有效的距离值")
        return
    
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
        # ✅ 百分位数正规化（推荐）
        percentile_low = np.percentile(distances_valid, 2)
        percentile_high = np.percentile(distances_valid, 98)
        print(f"    使用百分位数正规化: 2%-98% = [{percentile_low:.2f}, {percentile_high:.2f}]")
        
        if percentile_high > percentile_low:
            distances_normalized[valid_mask] = (distances[valid_mask] - percentile_low) / (percentile_high - percentile_low)
            distances_normalized[valid_mask] = np.clip(distances_normalized[valid_mask], 0, 1)
        else:
            distances_normalized[valid_mask] = 0.5
    
    elif normalize_method == "log":
        # ✅ 对数正规化（适合跨度大的数据）
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
    
    # 使用连续色映射
    cmap = cm.get_cmap(cmap_name)
    colors = cmap(distances_normalized)[:, :3]
    colors[~valid_mask] = [0.5, 0.5, 0.5]  # 无效点灰色
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 显示点云
    print(f"生成3D点云可视化（{normalize_method}正规化）...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"{title}\n范围={dist_min:.2f}~{dist_max:.2f}m | 中位数={dist_median:.2f}m\n({normalize_method}正规化)",
        width=1200,
        height=800
    )
