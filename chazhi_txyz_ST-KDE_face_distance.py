from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


PROJECT_ROOT = Path(__file__).resolve().parent

# Input files
COORD_CSV = PROJECT_ROOT / "data" / "7303_xyzet_rotated-face_distance.csv"
TIME_CSV = PROJECT_ROOT / "data" / "7303_all.csv"
OUTPUT_ROOT = PROJECT_ROOT / "input_txyz" / "face_distance"

# Window and rendering settings
GRID_SIZE = 256
WINDOW_HOURS = 24.0
SLIDE_HOURS = 12.0
PHYSICAL_SIGMA_T_HOURS = 12.0
PHYSICAL_SIGMA_S_METERS = {
    "x": 68.18,
    "y": 30.11,
    "z": 7.56,
}
HISTORY_TRUNCATE_HOURS = PHYSICAL_SIGMA_T_HOURS * 3.0

# This threshold is for reporting/auditing only. It does not remove events.
LARGE_DISTANCE_AUDIT_M = 500.0

CHANNEL_ORDER = ("ahead", "behind")


def render_gaussian_aniso(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    grid_size: int = GRID_SIZE,
    sigma_u: float = 8.0,
    sigma_v: float = 16.0,
) -> np.ndarray:
    """Render a nonnegative anisotropic Gaussian event-density field."""
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    if not (len(u) == len(v) == len(weights)):
        raise ValueError("u、v、weights 的长度必须相同。")
    if sigma_u <= 0 or sigma_v <= 0:
        raise ValueError("sigma_u 和 sigma_v 必须大于 0。")

    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(weights)
    u = u[valid]
    v = v[valid]
    weights = weights[valid]

    xp = cp if HAS_CUPY else np
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    weights_xp = xp.asarray(weights, dtype=xp.float32)

    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)

    for i in range(len(u)):
        if weights[i] == 0:
            continue
        dist_sq_u = (x_grid - u_pix[i]) ** 2 / (2.0 * sigma_u**2)
        dist_sq_v = (y_grid - v_pix[i]) ** 2 / (2.0 * sigma_v**2)
        canvas += weights_xp[i] * xp.exp(-(dist_sq_u + dist_sq_v))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas


def save_density_heatmap(image: np.ndarray, path: Path) -> None:
    """Save a nonnegative density visualization; the NPY values are unchanged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.nan_to_num(
        np.asarray(image, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    max_value = float(np.max(image)) if image.size else 0.0
    if max_value > 0:
        normalized = np.clip(image / max_value, 0.0, 1.0)
    else:
        normalized = np.zeros_like(image, dtype=np.float32)

    color_image = plt.get_cmap("magma")(normalized)
    image_uint8 = (color_image[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def load_data(
    coord_csv: Path,
    time_csv: Path,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    """Load events, attach timestamps, normalize coordinates, and validate distance."""
    if not coord_csv.is_file():
        raise FileNotFoundError(f"找不到坐标及工作面距离 CSV：{coord_csv}")
    if not time_csv.is_file():
        raise FileNotFoundError(f"找不到时间 CSV：{time_csv}")

    coords = pd.read_csv(coord_csv)
    coords.columns = coords.columns.str.strip().str.lower()

    try:
        time_data = pd.read_csv(time_csv, encoding="gb2312")
    except UnicodeDecodeError:
        time_data = pd.read_csv(time_csv, encoding="utf-8")
    time_data.columns = time_data.columns.str.strip()

    required_coord_cols = {"x", "y", "z", "dist_to_face_m"}
    missing_cols = required_coord_cols - set(coords.columns)
    if missing_cols:
        raise ValueError(f"距离坐标 CSV 缺少字段：{sorted(missing_cols)}")
    if not {"日期", "时间"}.issubset(time_data.columns):
        raise ValueError("时间 CSV 缺少“日期”或“时间”列。")
    if len(coords) != len(time_data):
        raise ValueError(
            "距离坐标 CSV 与时间 CSV 行数不同，不能按行对齐：\n"
            f"  coords: {len(coords)}\n  time: {len(time_data)}"
        )

    datetime_values = pd.to_datetime(
        time_data["日期"].astype(str).str.strip()
        + " "
        + time_data["时间"].astype(str).str.strip(),
        errors="coerce",
    )

    df = coords.copy()
    df["_datetime"] = datetime_values.to_numpy()
    for col in ["x", "y", "z", "dist_to_face_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coord_values = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    valid_coords = df["_datetime"].notna().to_numpy() & np.isfinite(
        coord_values
    ).all(axis=1)
    coord_df = df.loc[valid_coords].copy()
    if coord_df.empty:
        raise ValueError("清理无效时间和坐标后没有可用事件。")

    # Use the same spatial normalization convention as the coal_level script.
    spatial_ranges: dict[str, float] = {}
    for col in ["x", "y", "z"]:
        coord_min = float(coord_df[col].min())
        coord_max = float(coord_df[col].max())
        spatial_ranges[col] = max(coord_max - coord_min, 1e-8)
        coord_df[f"{col}_norm"] = (coord_df[col] - coord_min) / spatial_ranges[col]

    distance_values = coord_df["dist_to_face_m"].to_numpy(dtype=np.float64)
    valid_distance = np.isfinite(distance_values)
    missing_distance_count = int((~valid_distance).sum())

    # 保留有效时间和坐标的全部事件，让窗口时间轴不受距离缺失影响。
    return coord_df, spatial_ranges, missing_distance_count



def generate_windows(
    df: pd.DataFrame,
) -> list[tuple[pd.DataFrame, str, pd.Timestamp]]:
    """Create labeled sliding windows, retaining exact start timestamps."""
    df = df.sort_values("_datetime").reset_index(drop=True)
    current_time = pd.Timestamp(df["_datetime"].min())
    end_time = pd.Timestamp(df["_datetime"].max())
    windows = []

    while current_time <= end_time:
        win_end = current_time + timedelta(hours=WINDOW_HOURS)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        label = current_time.strftime("%Y-%m-%d_%H")
        windows.append((df.loc[mask].copy(), label, current_time))
        current_time += timedelta(hours=SLIDE_HOURS)

    return windows


def save_matrix(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(image, dtype=np.float32))


def main() -> None:
    df, spatial_ranges, missing_distance_count = load_data(COORD_CSV, TIME_CSV)
    windows = generate_windows(df)

    distance = df["dist_to_face_m"].to_numpy(dtype=np.float64)
    ahead_count = int((distance > 0).sum())
    behind_count = int((distance < 0).sum())
    at_face_count = int((distance == 0).sum())
    quantiles = pd.Series(distance).quantile([0.5, 0.9, 0.95, 0.99, 1.0])
    large_distance_count = int((np.abs(distance) > LARGE_DISTANCE_AUDIT_M).sum())

    print(f"有效工作面距离事件：{len(df)}")
    print(f"无效/缺失距离（在有效坐标事件中）：{missing_distance_count}")
    print(f"面前（正值）：{ahead_count}；面后（负值）：{behind_count}；零距离：{at_face_count}")
    print(f"距离分位数（米）：\n{quantiles.to_string()}")
    print(
        f"|dist_to_face_m| > {LARGE_DISTANCE_AUDIT_M:g} m 的事件："
        f"{large_distance_count}（仅报告，不自动剔除）"
    )
    print(f"输出目录：{OUTPUT_ROOT}")

    for _, label, win_start_dt in windows:
        win_end_dt = win_start_dt + timedelta(hours=WINDOW_HOURS)
        truncate_threshold_dt = win_start_dt - timedelta(
            hours=HISTORY_TRUNCATE_HOURS
        )

        history_mask = (
            (df["_datetime"] >= truncate_threshold_dt)
            & (df["_datetime"] < win_end_dt)
        )
        combined_df = df.loc[history_mask]

        x = combined_df["x_norm"].to_numpy(dtype=np.float32)
        y = combined_df["y_norm"].to_numpy(dtype=np.float32)
        z = combined_df["z_norm"].to_numpy(dtype=np.float32)
        distance = combined_df["dist_to_face_m"].to_numpy(dtype=np.float64)

        t_raw = (
            combined_df["_datetime"].astype("int64").to_numpy(dtype=np.float64)
            / 1e9
        )
        win_t_start = win_start_dt.value / 1e9
        win_t_end = win_end_dt.value / 1e9
        t = (t_raw - win_t_start) / (win_t_end - win_t_start)

        delta_t_hours = np.maximum((win_t_end - t_raw) / 3600.0, 0.0)
        time_weight = np.exp(
            -(delta_t_hours**2) / (2.0 * PHYSICAL_SIGMA_T_HOURS**2)
        ).astype(np.float32)

        # Each event contributes unit weight. Distance magnitude is not an event weight.
        event_weight = np.ones(len(combined_df), dtype=np.float32)
        projections = [
            ("tx", t, x, "t", "x"),
            ("ty", t, y, "t", "y"),
            ("tz", t, z, "t", "z"),
            ("xy", x, y, "x", "y"),
            ("xz", x, z, "x", "z"),
            ("yz", y, z, "y", "z"),
        ]

        sigma_t_pixels = (
            PHYSICAL_SIGMA_T_HOURS / WINDOW_HOURS
        ) * GRID_SIZE
        channel_masks = {
            "ahead": distance > 0,
            "behind": distance < 0,
            "at_face": distance == 0,
        }
        rendered_by_projection: dict[str, dict[str, np.ndarray]] = {}

        for proj_name, u_vals, v_vals, col_u, col_v in projections:
            if col_u == "t":
                sigma_u_pixels = sigma_t_pixels
            else:
                sigma_u_pixels = (
                    PHYSICAL_SIGMA_S_METERS[col_u] / spatial_ranges[col_u]
                ) * GRID_SIZE

            if col_v == "t":
                sigma_v_pixels = sigma_t_pixels
            else:
                sigma_v_pixels = (
                    PHYSICAL_SIGMA_S_METERS[col_v] / spatial_ranges[col_v]
                ) * GRID_SIZE

            # Match coal_level behavior: temporal projections are not decayed;
            # spatial-only projections decay older events by their age.
            if "t" in (col_u, col_v):
                weights = event_weight
            else:
                weights = event_weight * time_weight

            channel_images: dict[str, np.ndarray] = {}
            for channel_name, mask in channel_masks.items():
                target = render_gaussian_aniso(
                    u_vals[mask],
                    v_vals[mask],
                    weights[mask],
                    grid_size=GRID_SIZE,
                    sigma_u=sigma_u_pixels,
                    sigma_v=sigma_v_pixels,
                )
                channel_images[channel_name] = target.astype(np.float32)

                channel_dir = OUTPUT_ROOT / "matrix_npy" / channel_name / proj_name
                save_matrix(target, channel_dir / f"{label}.npy")

                vis_dir = OUTPUT_ROOT / "cov_face_vis" / channel_name / proj_name
                save_density_heatmap(target, vis_dir / f"{label}.png")

            # Primary ConvLSTM input: channel 0=ahead, channel 1=behind.
            stacked = np.stack(
                [channel_images[name] for name in CHANNEL_ORDER], axis=0
            )
            stacked_dir = OUTPUT_ROOT / "matrix_npy" / "stacked" / proj_name
            save_matrix(stacked, stacked_dir / f"{label}.npy")

            rendered_by_projection[proj_name] = channel_images

        print(
            f"完成窗口 {label}：历史事件 {len(combined_df)} 个，"
            f"ahead={(distance > 0).sum()}，behind={(distance < 0).sum()}，"
            f"at_face={(distance == 0).sum()}。"
        )

    print("工作面前/后事件密度场生成完成。")
    print(f"堆叠矩阵通道顺序：{CHANNEL_ORDER}；形状为 (2, {GRID_SIZE}, {GRID_SIZE})。")
    print("at_face 矩阵单独保存，不包含在两通道堆叠矩阵中。")


if __name__ == "__main__":
    if HAS_CUPY:
        print("检测到 CuPy，将使用 GPU 加速高斯渲染。")
    main()
