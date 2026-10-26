from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


# 项目路径：不依赖命令行当前工作目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "img" / "GongZuoMian33.png"
COORD_CSV = PROJECT_ROOT / "data" / "7303_xyzet_rotated.csv"

# 与 data/data_trans.py 保持一致
IMAGE_WIDTH = 3188
IMAGE_HEIGHT = 2363
WORKPLACE_X = (20400911.0, 20402820.0)
WORKPLACE_Y = (3915809.0, 3917226.0)

# 红色 HSV 阈值（OpenCV 的 H 范围是 0..179）
RED1_LOWER = np.array([0, 50, 50], dtype=np.uint8)
RED1_UPPER = np.array([10, 255, 255], dtype=np.uint8)
RED2_LOWER = np.array([170, 50, 50], dtype=np.uint8)
RED2_UPPER = np.array([179, 255, 255], dtype=np.uint8)

# 连通域过滤：增大可过滤更多文字/噪点，但可能丢失较短的断层线段
MIN_COMPONENT_AREA = 8
MIN_COMPONENT_SPAN = 8

# 默认不做形态学操作，避免腐蚀掉细红线；需要连接小断点时可设为 3 或 5
CLOSE_KERNEL_SIZE = 0

# 可视化设置
SHOW_PLOT = True
SAVE_PLOT = True
MAX_CONNECTORS = 250  # None 表示绘制全部最近点连线

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_image(path: Path) -> np.ndarray:
    """读取底图，并检查是否与坐标变换使用的尺寸一致。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到底图：{path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取底图：{path}")

    height, width = image.shape[:2]
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(
            f"底图尺寸为 {width}x{height}，但坐标变换配置是 "
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}。请确认使用的是同一张底图。"
        )
    return image


def extract_red_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """提取红色像素，并按连通域面积和跨度过滤小噪点。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    raw_mask = cv2.bitwise_or(mask1, mask2)

    if CLOSE_KERNEL_SIZE and CLOSE_KERNEL_SIZE > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
        )
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    line_mask = np.zeros_like(raw_mask)
    kept_components = 0

    for label_id in range(1, count):  # 跳过背景
        area = stats[label_id, cv2.CC_STAT_AREA]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        span = max(width, height)

        if area >= MIN_COMPONENT_AREA and span >= MIN_COMPONENT_SPAN:
            line_mask[labels == label_id] = 255
            kept_components += 1

    return raw_mask, line_mask, kept_components


def save_extraction_images(
    image: np.ndarray,
    raw_mask: np.ndarray,
    line_mask: np.ndarray,
    image_path: Path,
) -> None:
    """保存原始掩膜、过滤后掩膜和断层提取叠加图。"""
    raw_path = image_path.with_name(f"{image_path.stem}_red_mask_raw.png")
    mask_path = image_path.with_name(f"{image_path.stem}_red_mask.png")
    overlay_path = image_path.with_name(f"{image_path.stem}_red_line_overlay.png")

    overlay = image.copy()
    overlay[line_mask > 0] = (0, 255, 0)  # 绿色表示被保留的红色像素

    for path, output in (
        (raw_path, raw_mask),
        (mask_path, line_mask),
        (overlay_path, overlay),
    ):
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"无法保存图片：{path}")

    print(f"原始红色掩膜：{raw_path}")
    print(f"过滤后掩膜：  {mask_path}")
    print(f"提取叠加图：  {overlay_path}")


def load_microseismic_data(csv_path: Path) -> pd.DataFrame:
    """读取旋转后的坐标 CSV；其中 x、y 是反变换后的工作面坐标。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到坐标 CSV：{csv_path}")

    data = pd.read_csv(csv_path)
    #missing_columns = {"x", "y"} - set(data.columns)
    required_columns = {"x", "y", "z", "e", "t"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"坐标 CSV 缺少字段：{sorted(missing_columns)}")

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    valid = np.isfinite(data["x"].to_numpy()) & np.isfinite(data["y"].to_numpy())
    if not valid.any():
        raise ValueError("坐标 CSV 中没有有效的 x、y 坐标。")
    if not valid.all():
        print(f"警告：有 {(~valid).sum()} 行坐标无效，距离将留空。")

    return data


def calculate_distances(
    data: pd.DataFrame,
    line_mask: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """计算各事件到最近断层像素的距离，并返回绘图所需坐标。"""
    y_pixel, x_pixel = np.where(line_mask > 0)
    if len(x_pixel) == 0:
        raise RuntimeError(
            "过滤后没有红色像素。请检查掩膜图，并调小 MIN_COMPONENT_AREA / "
            "MIN_COMPONENT_SPAN 或调整 HSV 阈值。"
        )

    # 图像 y 轴向下，项目绘图坐标 y 轴向上；像素中心坐标加 0.5。
    fault_xy_pixel = np.column_stack(
        (
            x_pixel.astype(np.float64) + 0.5,
            IMAGE_HEIGHT - y_pixel.astype(np.float64) - 0.5,
        )
    )

    x_min, x_max = WORKPLACE_X
    y_min, y_max = WORKPLACE_Y
    scale_x = (x_max - x_min) / IMAGE_WIDTH
    scale_y = (y_max - y_min) / IMAGE_HEIGHT

    x_values = data["x"].to_numpy(dtype=np.float64)
    y_values = data["y"].to_numpy(dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)

    # 将旋转后 CSV 中的工作面坐标映射回 data_trans.py 使用的像素坐标。
    event_xy_pixel = np.column_stack(
        (
            (x_values[valid] - x_min) / (x_max - x_min) * IMAGE_WIDTH,
            (y_values[valid] - y_min) / (y_max - y_min) * IMAGE_HEIGHT,
        )
    )

    # 按 x、y 各自的米/像素比例建立 KDTree，距离单位为米。
    pixel_to_meter = np.array([scale_x, scale_y], dtype=np.float64)
    tree = cKDTree(fault_xy_pixel * pixel_to_meter)
    distance_m, nearest_indices = tree.query(
        event_xy_pixel * pixel_to_meter, k=1
    )

    nearest_fault_xy_pixel = fault_xy_pixel[nearest_indices]
    # 该像素距离对应“米制距离最近”的红色像素。
    # distance_pixel = np.linalg.norm(event_xy_pixel - nearest_fault_xy_pixel, axis=1)

    # data["dist_to_fault_pixels"] = np.nan
    # data["dist_to_fault_m"] = np.nan
    # data.loc[valid, "dist_to_fault_pixels"] = distance_pixel
    # data.loc[valid, "dist_to_fault_m"] = distance_m

    # return data, event_xy_pixel, nearest_fault_xy_pixel, distance_m
    
    # distance_m 是微震点到物理距离最近的提取红色像素中心的距离，单位为米。
    data["distance_to_nearest_red_line"] = np.nan
    data.loc[valid, "distance_to_nearest_red_line"] = distance_m

    return data, event_xy_pixel, nearest_fault_xy_pixel, distance_m



def plot_distances(
    image: np.ndarray,
    line_mask: np.ndarray,
    event_xy_pixel: np.ndarray,
    nearest_fault_xy_pixel: np.ndarray,
    distance_m: np.ndarray,
    output_path: Path,
) -> None:
    """在底图上绘制断层、全部微震点及最近距离。"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)

    # extent 使图像坐标与 data_trans.py 的 x=[0,W], y=[0,H] 对齐。
    ax.imshow(
        image_rgb,
        origin="upper",
        extent=(0, IMAGE_WIDTH, 0, IMAGE_HEIGHT),
        zorder=0,
    )

    # 用半透明青色强调从底图提取出的断层像素。
    fault_overlay = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 4), dtype=np.float32)
    #fault_overlay[line_mask > 0] = (0.0, 1.0, 1.0, 0.9)
    fault_overlay[line_mask > 0] = (1.0, 0.0, 0.0, 0.9)
    ax.imshow(
        fault_overlay,
        origin="upper",
        extent=(0, IMAGE_WIDTH, 0, IMAGE_HEIGHT),
        interpolation="nearest",
        zorder=1,
    )

    point_count = len(event_xy_pixel)
    if MAX_CONNECTORS is None or MAX_CONNECTORS >= point_count:
        connector_indices = np.arange(point_count)
    else:
        connector_count = max(0, int(MAX_CONNECTORS))
        connector_indices = np.linspace(
            0, point_count - 1, connector_count, dtype=int
        ) if connector_count else np.array([], dtype=int)

    if len(connector_indices):
        segments = np.stack(
            (
                event_xy_pixel[connector_indices],
                nearest_fault_xy_pixel[connector_indices],
            ),
            axis=1,
        )
        connectors = LineCollection(
            segments,
            colors="#ffe66d",
            linewidths=0.55,
            alpha=0.45,
            zorder=2,
        )
        ax.add_collection(connectors)

    scatter = ax.scatter(
        event_xy_pixel[:, 0],
        event_xy_pixel[:, 1],
        c=distance_m,
        cmap="turbo",
        s=14,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.15,
        zorder=3,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.88)
    colorbar.set_label("到最近红色断层线的距离 (m)")

    ax.set_xlim(0, IMAGE_WIDTH)
    ax.set_ylim(0, IMAGE_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("X (像素)")
    ax.set_ylabel("Y (像素，向上为正)")
    ax.set_title("微震点到红色断层线的最近距离")
    ax.grid(False)

    connector_note = (
        f"黄色连线：显示全部 {len(connector_indices)} 条最近距离连线"
        if len(connector_indices) == point_count
        else f"黄色连线：抽样显示 {len(connector_indices)} / {point_count} 条最近距离连线"
    )
    stats_text = (
        f"事件点：{point_count}\n"
        f"距离范围：{np.min(distance_m):.2f} - {np.max(distance_m):.2f} m\n"
        f"平均距离：{np.mean(distance_m):.2f} m\n"
        f"{connector_note}"
    )
    ax.text(
        0.015,
        0.985,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 5},
        zorder=4,
    )

    if SAVE_PLOT:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f"距离可视化图：{output_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


def main() -> None:
    image = read_image(IMAGE_PATH)
    raw_mask, line_mask, kept_components = extract_red_mask(image)
    save_extraction_images(image, raw_mask, line_mask, IMAGE_PATH)

    print(f"保留的红色连通域：{kept_components}")
    print(f"保留的红色像素：{np.count_nonzero(line_mask)}")

    data = load_microseismic_data(COORD_CSV)
    data, event_xy_pixel, nearest_fault_xy_pixel, distance_m = calculate_distances(
        data, line_mask
    )

    output_dir = PROJECT_ROOT / "script_update" / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "7303_red_line_distances.csv"
    output_plot = output_csv.with_suffix(".png")

    output_columns = ["x", "y", "z", "e", "t", "distance_to_nearest_red_line"]
    data.loc[:, output_columns].to_csv(output_csv, index=False)

    print(f"距离结果 CSV：{output_csv}")
    print("距离结果示例：")
    print(data.loc[:, output_columns].head().to_string(index=False))

    print(
        "比例尺："
        f"x={(WORKPLACE_X[1] - WORKPLACE_X[0]) / IMAGE_WIDTH:.6f} 米/像素，"
        f"y={(WORKPLACE_Y[1] - WORKPLACE_Y[0]) / IMAGE_HEIGHT:.6f} 米/像素"
    )

    plot_distances(
        image=image,
        line_mask=line_mask,
        event_xy_pixel=event_xy_pixel,
        nearest_fault_xy_pixel=nearest_fault_xy_pixel,
        distance_m=distance_m,
        output_path=output_plot,
    )


if __name__ == "__main__":
    main()
