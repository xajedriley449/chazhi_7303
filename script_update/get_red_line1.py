from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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

MIN_COMPONENT_AREA = 8
MIN_COMPONENT_SPAN = 8
CLOSE_KERNEL_SIZE = 0

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到底图：{path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取底图：{path}")

    height, width = image.shape[:2]
    if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
        raise ValueError(f"底图尺寸不匹配。")
    return image


def extract_red_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
    mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
    raw_mask = cv2.bitwise_or(mask1, mask2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    line_mask = np.zeros_like(raw_mask)
    kept_components = 0

    for label_id in range(1, count):
        area = stats[label_id, cv2.CC_STAT_AREA]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        span = max(width, height)

        if area >= MIN_COMPONENT_AREA and span >= MIN_COMPONENT_SPAN:
            line_mask[labels == label_id] = 255
            kept_components += 1

    return raw_mask, line_mask, kept_components


# def verify_extraction_purity(image: np.ndarray, line_mask: np.ndarray) -> None:
#     """新增的纯净度检验可视化函数"""
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # 创建一个纯黑背景图像
#     pure_extraction = np.zeros_like(image_rgb)
#     # 仅将提取到的（掩膜为白色的）像素点从原图拷贝到黑底上
#     pure_extraction[line_mask > 0] = image_rgb[line_mask > 0]

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    
#     ax1.imshow(image_rgb)
#     ax1.set_title("原始底图 (包含所有颜色和干扰项)")
#     ax1.axis("off")

#     ax2.imshow(pure_extraction)
#     ax2.set_title("纯净度提取检验 (纯黑背景 + 提取像素)")
#     ax2.axis("off")

#     plt.show()
def verify_extraction_purity(image: np.ndarray, line_mask: np.ndarray) -> None:
    """新增的纯净度检验可视化函数（纯白背景）"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建一个纯白背景图像 (用 255 填充)
    pure_extraction = np.full_like(image_rgb, 255)
    
    # 仅将提取到的（掩膜为白色的）像素点从原图拷贝到白底上
    pure_extraction[line_mask > 0] = image_rgb[line_mask > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    
    ax1.imshow(image_rgb)
    ax1.set_title("原始底图 (包含所有颜色和干扰项)")
    ax1.axis("off")

    ax2.imshow(pure_extraction)
    ax2.set_title("纯净度提取检验 (纯白背景 + 提取像素)")
    ax2.axis("off")

    plt.show()


def load_microseismic_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到坐标 CSV：{csv_path}")

    data = pd.read_csv(csv_path)
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    return data


def calculate_distances(
    data: pd.DataFrame,
    line_mask: np.ndarray,
) -> pd.DataFrame:
    y_pixel, x_pixel = np.where(line_mask > 0)
    
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

    event_xy_pixel = np.column_stack(
        (
            (x_values[valid] - x_min) / (x_max - x_min) * IMAGE_WIDTH,
            (y_values[valid] - y_min) / (y_max - y_min) * IMAGE_HEIGHT,
        )
    )

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
    
    red_pixel_count = int(np.count_nonzero(line_mask))
    print(f"保留的红色连通域：{kept_components}")
    print(f"保留的红色像素：{red_pixel_count}")

    # 调用新增的可视化函数
    verify_extraction_purity(image, line_mask)

    data = load_microseismic_data(COORD_CSV)
    data = calculate_distances(data, line_mask)

    output_csv = COORD_CSV.with_name(
        f"{COORD_CSV.stem}_with_fault_dist.csv"
    )
    data.to_csv(output_csv, index=False)
    print(f"距离结果已保存：{output_csv}")


if __name__ == "__main__":
    main()