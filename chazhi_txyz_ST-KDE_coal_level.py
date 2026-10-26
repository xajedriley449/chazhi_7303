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


def render_gaussian_aniso(
    u: np.ndarray,
    v: np.ndarray,
    values: np.ndarray,
    grid_size: int = 256,
    sigma_u: float = 8.0,
    sigma_v: float = 16.0,
) -> np.ndarray:
    """Render a signed, anisotropic Gaussian sum on a square grid."""
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if not (len(u) == len(v) == len(values)):
        raise ValueError("u、v、values 的长度必须相同。")
    if sigma_u <= 0 or sigma_v <= 0:
        raise ValueError("sigma_u 和 sigma_v 必须大于 0。")

    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(values)
    u = u[valid]
    v = v[valid]
    values = values[valid]

    xp = cp if HAS_CUPY else np
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    values_xp = xp.asarray(values, dtype=xp.float32)

    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)

    # Signed values are intentional: negative coal_level values must remain negative.
    for i in range(len(u_pix)):
        if values[i] == 0:
            continue
        dist_sq_u = (x_grid - u_pix[i]) ** 2 / (2.0 * sigma_u**2)
        dist_sq_v = (y_grid - v_pix[i]) ** 2 / (2.0 * sigma_v**2)
        canvas += values_xp[i] * xp.exp(-(dist_sq_u + dist_sq_v))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas


def save_signed_heatmap(image: np.ndarray, path: Path) -> None:
    """Save a diverging visualization with zero at the color midpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = float(np.max(np.abs(image))) if image.size else 0.0
    if max_abs > 0:
        normalized = 0.5 + 0.5 * np.clip(image / max_abs, -1.0, 1.0)
    else:
        normalized = np.full_like(image, 0.5, dtype=np.float32)

    color_image = plt.get_cmap("coolwarm")(normalized)
    image_uint8 = (color_image[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def load_data(
    coord_csv: Path,
    time_csv: Path,
) -> tuple[pd.DataFrame, dict[str, float], int]:
    """Load coal-level events, attach timestamps, and normalize coordinates."""
    if not coord_csv.is_file():
        raise FileNotFoundError(f"找不到坐标及煤层数据 CSV：{coord_csv}")
    if not time_csv.is_file():
        raise FileNotFoundError(f"找不到时间数据 CSV：{time_csv}")

    coords = pd.read_csv(coord_csv)
    coords.columns = coords.columns.str.strip().str.lower()

    time_data = pd.read_csv(time_csv, encoding="gb2312")
    time_data.columns = time_data.columns.str.strip()

    required_coord_cols = {"x", "y", "z", "coal_level"}
    missing_cols = required_coord_cols - set(coords.columns)
    if missing_cols:
        raise ValueError(f"煤层坐标 CSV 缺少字段：{sorted(missing_cols)}")
    if not {"日期", "时间"}.issubset(time_data.columns):
        raise ValueError("时间 CSV 缺少 '日期' 或 '时间' 列。")
    if len(coords) != len(time_data):
        raise ValueError(
            f"煤层坐标 CSV 与时间 CSV 行数不同：{len(coords)} != {len(time_data)}。"
        )

    datetime_values = pd.to_datetime(
        time_data["日期"].astype(str).str.strip()
        + " "
        + time_data["时间"].astype(str).str.strip(),
        errors="coerce",
    )

    df = coords.copy()
    df["_datetime"] = datetime_values.to_numpy()
    for col in ["x", "y", "z", "coal_level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coord_values = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    valid_coords = df["_datetime"].notna().to_numpy() & np.isfinite(
        coord_values
    ).all(axis=1)
    coord_df = df.loc[valid_coords].copy()
    if coord_df.empty:
        raise ValueError("清理无效时间和坐标后没有可用事件。")

    spatial_ranges: dict[str, float] = {}
    for col in ["x", "y", "z"]:
        coord_min = float(coord_df[col].min())
        coord_max = float(coord_df[col].max())
        spatial_ranges[col] = max(coord_max - coord_min, 1e-8)
        coord_df[f"{col}_norm"] = (coord_df[col] - coord_min) / spatial_ranges[col]

    coal_values = coord_df["coal_level"].to_numpy(dtype=np.float64)
    valid_coal = np.isfinite(coal_values)
    missing_coal_count = int((~valid_coal).sum())
    df = coord_df.loc[valid_coal].copy()
    if df.empty:
        raise ValueError("有效坐标事件中没有可用的 coal_level 值。")

    max_abs_level = float(np.max(np.abs(df["coal_level"].to_numpy(dtype=np.float64))))
    if max_abs_level > 0:
        df["coal_level_norm"] = np.clip(
            df["coal_level"] / max_abs_level, -1.0, 1.0
        )
    else:
        df["coal_level_norm"] = 0.0

    return df, spatial_ranges, missing_coal_count


def generate_windows(
    df: pd.DataFrame,
    window_hours: float = 24.0,
    slide_hours: float = 12.0,
) -> list[tuple[pd.DataFrame, str, pd.Timestamp]]:
    """Create labeled sliding windows, retaining exact start timestamps."""
    df = df.sort_values("_datetime").reset_index(drop=True)
    current_time = pd.Timestamp(df["_datetime"].min())
    end_time = pd.Timestamp(df["_datetime"].max())
    windows = []

    while current_time <= end_time:
        win_end = current_time + timedelta(hours=window_hours)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        label = current_time.strftime("%Y-%m-%d_%H")
        windows.append((df.loc[mask].copy(), label, current_time))
        current_time += timedelta(hours=slide_hours)

    return windows


def main() -> None:
    coord_csv = PROJECT_ROOT / "data" / "7303_xyzet_rotated-coal_level.csv"
    time_csv = PROJECT_ROOT / "data" / "7303_all.csv"
    output_root = PROJECT_ROOT / "input_txyz" / "coal_level"

    grid_size = 256
    window_hours = 24.0
    slide_hours = 12.0

    physical_sigma_t_hours = 12.0
    physical_sigma_s_meters = {
        "x": 68.18,
        "y": 30.11,
        "z": 7.56,
    }
    history_truncate_hours = physical_sigma_t_hours * 3.0

    df, spatial_ranges, missing_coal_count = load_data(coord_csv, time_csv)
    windows = generate_windows(df, window_hours, slide_hours)

    print(
        f"有效 coal_level 事件 {len(df)} 个，缺失/无效煤层值 "
        f"{missing_coal_count} 个；共生成 {len(windows)} 个窗口。"
    )
    print(f"输出目录：{output_root}")

    for win_df, label, win_start_dt in windows:
        win_end_dt = win_start_dt + timedelta(hours=window_hours)
        truncate_threshold_dt = win_start_dt - timedelta(
            hours=history_truncate_hours
        )

        # Select each source event once, including recent history and this window.
        history_mask = (
            (df["_datetime"] >= truncate_threshold_dt)
            & (df["_datetime"] < win_end_dt)
        )
        combined_df = df.loc[history_mask]
        if combined_df.empty:
            print(f"跳过空历史窗口：{label}")
            continue

        print(
            f"处理窗口 {label}：当前窗口 {len(win_df)} 个点，"
            f"近期历史共 {len(combined_df)} 个点。"
        )

        x = combined_df["x_norm"].to_numpy(dtype=np.float32)
        y = combined_df["y_norm"].to_numpy(dtype=np.float32)
        z = combined_df["z_norm"].to_numpy(dtype=np.float32)
        coal_level = combined_df["coal_level_norm"].to_numpy(dtype=np.float32)

        # Use a timezone-independent epoch representation for relative time values.
        t_raw = combined_df["_datetime"].astype("int64").to_numpy(dtype=np.float64) / 1e9
        win_t_start = win_start_dt.value / 1e9
        win_t_end = win_end_dt.value / 1e9
        t = (t_raw - win_t_start) / (win_t_end - win_t_start)

        delta_t_hours = np.maximum((win_t_end - t_raw) / 3600.0, 0.0)
        time_weight = np.exp(
            -(delta_t_hours**2) / (2.0 * physical_sigma_t_hours**2)
        ).astype(np.float32)

        # Keep temporal projections un-decayed, matching the red_cov renderer.
        # Apply temporal decay to spatial-only projections.
        coal_spatial = coal_level * time_weight
        projections = [
            ("tx", t, x, "t", "x", coal_level),
            ("ty", t, y, "t", "y", coal_level),
            ("tz", t, z, "t", "z", coal_level),
            ("xy", x, y, "x", "y", coal_spatial),
            ("xz", x, z, "x", "z", coal_spatial),
            ("yz", y, z, "y", "z", coal_spatial),
        ]

        sigma_t_pixels = (physical_sigma_t_hours / window_hours) * grid_size

        for proj_name, u_vals, v_vals, col_u, col_v, values in projections:
            if col_u == "t":
                sigma_u_pixels = sigma_t_pixels
            else:
                sigma_u_pixels = (
                    physical_sigma_s_meters[col_u] / spatial_ranges[col_u]
                ) * grid_size

            sigma_v_pixels = (
                physical_sigma_s_meters[col_v] / spatial_ranges[col_v]
            ) * grid_size

            target = render_gaussian_aniso(
                u_vals,
                v_vals,
                values,
                grid_size=grid_size,
                sigma_u=sigma_u_pixels,
                sigma_v=sigma_v_pixels,
            )

            npy_dir = output_root / "matrix_npy" / proj_name
            npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_dir / f"{label}.npy", target.astype(np.float32))

            vis_dir = output_root / "cov_level_vis" / proj_name
            save_signed_heatmap(target, vis_dir / f"{label}.png")

    print("煤层 coal_level 协变量场生成完成。")


if __name__ == "__main__":
    if HAS_CUPY:
        print("检测到 CuPy，将使用 GPU 加速高斯渲染。")
    main()
