import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Optional GPU acceleration, matching the renderer in chazhi_txyz_ST-KDE.py.
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


def render_gaussian_aniso(
    u: np.ndarray,
    v: np.ndarray,
    values: np.ndarray,
    grid_size: int = 256,
    sigma_u: float = 8.0,
    sigma_v: float = 16.0,
) -> np.ndarray:
    """Accumulate event values with the same anisotropic Gaussian kernel."""
    xp = cp if HAS_CUPY else np

    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    values = xp.asarray(values, dtype=xp.float32)

    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)

    for i in range(len(u_pix)):
        if values[i] <= 0:
            continue
        dist_sq_u = (x_grid - u_pix[i]) ** 2 / (2 * sigma_u ** 2)
        dist_sq_v = (y_grid - v_pix[i]) ** 2 / (2 * sigma_v ** 2)
        canvas += values[i] * xp.exp(-(dist_sq_u + dist_sq_v))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas


def save_heatmap(image: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.asarray(image, dtype=np.float32)

    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    else:
        image = np.zeros_like(image)

    color_image = plt.get_cmap("viridis")(np.clip(image, 0, 1))
    image_uint8 = (color_image[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def load_data(coord_csv: str, time_csv: str, red_dist_csv: str) -> tuple[pd.DataFrame, dict]:
    """Load events, attach timestamps and red-fault distances, and normalize fields."""
    coords = pd.read_csv(coord_csv)
    coords.columns = coords.columns.str.strip().str.lower()

    red = pd.read_csv(red_dist_csv)
    red.columns = red.columns.str.strip().str.lower()

    time_data = pd.read_csv(time_csv, encoding="gb2312")
    time_data.columns = time_data.columns.str.strip()
    if not {"日期", "时间"}.issubset(time_data.columns):
        raise ValueError("时间数据缺少 '日期' 或 '时间' 列。")

    key_cols = ["x", "y", "z", "e", "t"]
    required_red_cols = key_cols + ["distance_to_nearest_red_line"]
    missing_coords = [col for col in key_cols if col not in coords.columns]
    missing_red = [col for col in required_red_cols if col not in red.columns]
    if missing_coords or missing_red:
        raise ValueError(
            f"CSV 缺少必要列；坐标 CSV: {missing_coords}, 红线 CSV: {missing_red}"
        )

    if len(coords) != len(time_data):
        raise ValueError(
            f"坐标 CSV 与时间 CSV 行数不同: {len(coords)} != {len(time_data)}"
        )
    if len(coords) != len(red):
        raise ValueError(
            f"坐标 CSV 与红线距离 CSV 行数不同: {len(coords)} != {len(red)}"
        )

    # Check row-wise correspondence before assigning distances by row position.
    for col in key_cols:
        coord_values = pd.to_numeric(coords[col], errors="coerce").to_numpy(dtype=float)
        red_values = pd.to_numeric(red[col], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(
            coord_values, red_values, rtol=0.0, atol=1e-6, equal_nan=True
        ):
            raise ValueError(
                f"坐标 CSV 与红线距离 CSV 的 {col} 列行顺序或数值不一致。"
            )

    datetime_values = pd.to_datetime(
        time_data["日期"].astype(str).str.strip()
        + " "
        + time_data["时间"].astype(str).str.strip(),
        errors="coerce",
    )

    df = coords.copy()
    df["_datetime"] = datetime_values.to_numpy()
    df["distance_red"] = pd.to_numeric(
        red["distance_to_nearest_red_line"], errors="coerce"
    ).to_numpy()

    for col in ["x", "y", "z", "e", "t", "distance_red"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["_datetime", "x", "y", "z", "distance_red"]).copy()
    if df.empty:
        raise ValueError("清理无效时间、坐标或红线距离后没有可用事件。")

    red_min = df["distance_red"].min()
    red_max = df["distance_red"].max()
    df["dist_red_norm"] = (df["distance_red"] - red_min) / (
        red_max - red_min + 1e-8
    )

    spatial_ranges = {}
    for col in ["x", "y", "z"]:
        coord_min = df[col].min()
        coord_max = df[col].max()
        spatial_ranges[col] = coord_max - coord_min + 1e-8
        df[f"{col}_norm"] = (df[col] - coord_min) / spatial_ranges[col]

    return df, spatial_ranges


def generate_windows(
    df: pd.DataFrame,
    window_hours: float = 24.0,
    slide_hours: float = 12.0,
) -> list[tuple[pd.DataFrame, str]]:
    df = df.sort_values("_datetime").reset_index(drop=True)
    current_time = df["_datetime"].min()
    end_time = df["_datetime"].max()
    windows = []

    while current_time <= end_time:
        win_end = current_time + timedelta(hours=window_hours)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        label = current_time.strftime("%Y-%m-%d_%H")
        windows.append((df.loc[mask].copy(), label))
        current_time += timedelta(hours=slide_hours)

    return windows


def main() -> None:
    coord_csv = r"data/7303_xyzet_rotated.csv"
    time_csv = r"data/7303_all.csv"
    red_dist_csv = r"script_update/result/7303_red_line_distances.csv"
    out_red_cov = r"input_txyz/red_cov"

    grid_size = 256
    window_hours = 24.0
    slide_hours = 12.0

    # Keep these physical bandwidths identical to chazhi_txyz_ST-KDE.py.
    physical_sigma_t_hours = 12.0
    physical_sigma_s_meters = {
        "x": 68.18,
        "y": 30.11,
        "z": 7.56,
    }
    history_truncate_hours = physical_sigma_t_hours * 3.0

    df, spatial_ranges = load_data(coord_csv, time_csv, red_dist_csv)
    windows = generate_windows(df, window_hours, slide_hours)
    print(f"共生成 {len(windows)} 个窗口，输出目录: {out_red_cov}")

    history_dfs = []

    for win_df, label in windows:
        win_start_dt = pd.to_datetime(label, format="%Y-%m-%d_%H")
        win_end_dt = win_start_dt + timedelta(hours=window_hours)
        win_t_start = win_start_dt.timestamp()
        win_t_end = win_end_dt.timestamp()

        history_dfs.append(win_df)
        truncate_threshold_dt = win_start_dt - timedelta(
            hours=history_truncate_hours
        )
        history_dfs = [
            hist.loc[hist["_datetime"] >= truncate_threshold_dt]
            for hist in history_dfs
            if (hist["_datetime"] >= truncate_threshold_dt).any()
        ]

        combined_df = pd.concat(history_dfs, ignore_index=True)
        print(
            f"处理窗口 {label}: 当前窗口 {len(win_df)} 个点，"
            f"近期历史共 {len(combined_df)} 个点"
        )

        x = combined_df["x_norm"].to_numpy()
        y = combined_df["y_norm"].to_numpy()
        z = combined_df["z_norm"].to_numpy()
        dist_red = combined_df["dist_red_norm"].to_numpy()

        t_raw = combined_df["_datetime"].map(lambda value: value.timestamp()).to_numpy()
        t = (t_raw - win_t_start) / ((win_t_end - win_t_start) + 1e-8)

        delta_t_hours = np.maximum((win_t_end - t_raw) / 3600.0, 0.0)
        time_weight = np.exp(
            -(delta_t_hours ** 2) / (2 * physical_sigma_t_hours ** 2)
        )

        # For t-x/y/z projections, preserve the original script's unweighted
        # temporal covariate behavior. For x-y/x-z/y-z, apply its time decay.
        proximity_red_spatial = (1.0 - dist_red) * time_weight
        projections = [
            ("tx", t, x, "t", "x", 1.0 - dist_red),
            ("ty", t, y, "t", "y", 1.0 - dist_red),
            ("tz", t, z, "t", "z", 1.0 - dist_red),
            ("xy", x, y, "x", "y", proximity_red_spatial),
            ("xz", x, z, "x", "z", proximity_red_spatial),
            ("yz", y, z, "y", "z", proximity_red_spatial),
        ]

        sigma_t_pixels = (
            physical_sigma_t_hours / window_hours
        ) * grid_size

        for proj_name, u_vals, v_vals, col_u, col_v, red_values in projections:
            if col_u == "t":
                sigma_u_pixels = sigma_t_pixels
            else:
                sigma_u_pixels = (
                    physical_sigma_s_meters[col_u] / spatial_ranges[col_u]
                ) * grid_size

            sigma_v_pixels = (
                physical_sigma_s_meters[col_v] / spatial_ranges[col_v]
            ) * grid_size

            red_target = render_gaussian_aniso(
                u_vals,
                v_vals,
                red_values,
                grid_size=grid_size,
                sigma_u=sigma_u_pixels,
                sigma_v=sigma_v_pixels,
            )

            npy_dir = os.path.join(out_red_cov, "matrix_npy", proj_name)
            os.makedirs(npy_dir, exist_ok=True)
            np.save(os.path.join(npy_dir, f"{label}.npy"), red_target)

            vis_dir = os.path.join(out_red_cov, "cov_distance_vis", proj_name)
            save_heatmap(red_target, os.path.join(vis_dir, f"{label}.png"))

    print("红线距离协变量场生成完成。")


if __name__ == "__main__":
    if HAS_CUPY:
        print("检测到 CuPy，将使用 GPU 加速高斯渲染。")
    main()
