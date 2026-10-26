from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# 项目路径：不依赖当前命令行所在目录
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

# 连通域过滤：适当调大可过滤小文字/噪点，但可能丢失断裂的细线段
MIN_COMPONENT_AREA = 8
MIN_COMPONENT_SPAN = 8

# 默认不做形态学操作，避免腐蚀掉细红线。需要连接小断点时可设为 3。
CLOSE_KERNEL_SIZE = 0


def read_image(path: Path) -> np.ndarray:
    """读取底图并检查尺寸，尺寸必须与坐标变换时使用的像素范围一致。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到底图：{path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取底图：{path}")

    height, width = image.shape[:2]
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(
            f"底图尺寸为 {width}x{height}，但坐标变换配置是 "
            f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}。请确认图片与 data_trans.py 使用的是同一张图。"
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
    """保存原始红色掩膜、过滤后掩膜和便于人工检查的叠加图。"""
    raw_path = image_path.with_name(f"{image_path.stem}_red_mask_raw.png")
    mask_path = image_path.with_name(f"{image_path.stem}_red_mask.png")
    overlay_path = image_path.with_name(f"{image_path.stem}_red_line_overlay.png")

    overlay = image.copy()
    overlay[line_mask > 0] = (0, 255, 0)  # 用绿色标出保留的红色像素

    for path, output in (
        (raw_path, raw_mask),
        (mask_path, line_mask),
        (overlay_path, overlay),
    ):
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"无法保存图片：{path}")

    print(f"原始红色掩膜：{raw_path}")
    print(f"过滤后掩膜：  {mask_path}")
    print(f"红线叠加图：  {overlay_path}")


def load_microseismic_data(csv_path: Path) -> pd.DataFrame:
    """读取旋转后的坐标表。CSV 中 x、y 是反变换后的工作面坐标。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到坐标 CSV：{csv_path}")

    data = pd.read_csv(csv_path)
    missing_columns = {"x", "y"} - set(data.columns)
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
) -> pd.DataFrame:
    """在物理距离（米）度量下，计算每个事件到红线像素的最近距离。"""
    y_pixel, x_pixel = np.where(line_mask > 0)
    if len(x_pixel) == 0:
        raise RuntimeError(
            "过滤后没有红色像素。请检查原始掩膜，并调小 MIN_COMPONENT_AREA / "
            "MIN_COMPONENT_SPAN 或调整 HSV 阈值。"
        )

    # data_trans.py 使用像素坐标范围 x=[0, IMAGE_WIDTH], y=[0, IMAGE_HEIGHT]。
    # 红色像素用像素中心坐标；y 转成原脚本使用的向上为正坐标系。
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

    # 旋转后 CSV 中的工作面坐标，按 data_trans.py 的 MinMaxScaler 反向还原到像素坐标。
    event_xy_pixel = np.column_stack(
        (
            (x_values[valid] - x_min) / (x_max - x_min) * IMAGE_WIDTH,
            (y_values[valid] - y_min) / (y_max - y_min) * IMAGE_HEIGHT,
        )
    )

    # 分别使用 x/y 比例尺换算为米，再查询最近点，避免假设两个方向像素比例完全相同。
    pixel_to_meter = np.array([scale_x, scale_y], dtype=np.float64)
    tree = cKDTree(fault_xy_pixel * pixel_to_meter)
    distance_m, nearest_indices = tree.query(event_xy_pixel * pixel_to_meter, k=1)

    nearest_fault_xy_pixel = fault_xy_pixel[nearest_indices]
    distance_pixel = np.linalg.norm(event_xy_pixel - nearest_fault_xy_pixel, axis=1)

    data["dist_to_fault_pixels"] = np.nan
    data["dist_to_fault_m"] = np.nan
    data.loc[valid, "dist_to_fault_pixels"] = distance_pixel
    data.loc[valid, "dist_to_fault_m"] = distance_m
    return data


def main() -> None:
    image = read_image(IMAGE_PATH)
    raw_mask, line_mask, kept_components = extract_red_mask(image)
    save_extraction_images(image, raw_mask, line_mask, IMAGE_PATH)

    red_pixel_count = int(np.count_nonzero(line_mask))
    print(f"保留的红色连通域：{kept_components}")
    print(f"保留的红色像素：{red_pixel_count}")

    data = load_microseismic_data(COORD_CSV)
    data = calculate_distances(data, line_mask)

    output_csv = COORD_CSV.with_name(
        f"{COORD_CSV.stem}_with_fault_dist.csv"
    )
    data.to_csv(output_csv, index=False)

    print(f"距离结果已保存：{output_csv}")
    print("距离结果示例：")
    print(
        data[["x", "y", "dist_to_fault_pixels", "dist_to_fault_m"]]
        .head()
        .to_string(index=False)
    )
    print(f"比例尺：x={((WORKPLACE_X[1] - WORKPLACE_X[0]) / IMAGE_WIDTH):.6f} 米/像素，")
    print(f"        y={((WORKPLACE_Y[1] - WORKPLACE_Y[0]) / IMAGE_HEIGHT):.6f} 米/像素")


if __name__ == "__main__":
    main()
