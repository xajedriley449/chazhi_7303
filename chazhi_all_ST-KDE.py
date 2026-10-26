import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import QuantileTransformer

# 尝试导入 cupy 以加速高斯渲染
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

# ==========================================
# 模块1: 核心算法 - 高斯累加与分位数映射
# ==========================================
def render_gaussian_aniso(u: np.ndarray, v: np.ndarray, e: np.ndarray, grid_size: int = 256, sigma_u: float = 8.0, sigma_v: float = 16.0):
    """
    各向异性高斯累加插值 (Log-Gaussian KDE-Sum)
    允许在U轴(如时间)和V轴(如空间)设置不同的衰减带宽高斯参数。
    """
    xp = cp if HAS_CUPY else np
    
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    e_val = xp.asarray(e, dtype=xp.float32)
    
    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)
    
    for i in range(len(u_pix)):
        if e_val[i] <= 0: continue
        # 分别计算 U 轴和 V 轴的归一化距离的平方
        dist_sq_u = (x_grid - u_pix[i])**2 / (2 * sigma_u**2)
        dist_sq_v = (y_grid - v_pix[i])**2 / (2 * sigma_v**2)
        canvas += e_val[i] * xp.exp(-(dist_sq_u + dist_sq_v))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas

def render_gaussian_sum(u: np.ndarray, v: np.ndarray, e: np.ndarray, grid_size: int = 256, sigma: float = 12.0):
    """
    高斯累加插值 (Log-Gaussian KDE-Sum)
    """
    xp = cp if HAS_CUPY else np
    
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    e_val = xp.asarray(e, dtype=xp.float32)
    
    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)
    
    for i in range(len(u_pix)):
        if e_val[i] <= 0: continue
        dist_sq = (x_grid - u_pix[i])**2 + (y_grid - v_pix[i])**2
        canvas += e_val[i] * xp.exp(-dist_sq / (2 * sigma**2))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas

def apply_quantile_mapping(image_matrix: np.ndarray, background_threshold: float = 1e-5):
    """
    相对分位数映射 (Quantile CDF Mapping)
    """
    H, W = image_matrix.shape
    flat_img = image_matrix.flatten().reshape(-1, 1)
    
    valid_mask = flat_img > background_threshold
    valid_pixels = flat_img[valid_mask].reshape(-1, 1)
    
    if len(valid_pixels) < 10:
        return np.zeros_like(image_matrix)
        
    n_quantiles = min(1000, len(valid_pixels))
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    
    mapped_valid = qt.fit_transform(valid_pixels)
    
    mapped_img_flat = np.zeros_like(flat_img, dtype=np.float32)
    mapped_img_flat[valid_mask] = mapped_valid.flatten()
    
    return mapped_img_flat.reshape(H, W)


# ==========================================
# 模块2: 数据加载与时序窗口滑动
# ==========================================

def load_and_preprocess_data(coord_csv: str, time_csv: str, t_dist_csv: str, xyz_dist_csv: str):
    print("正在加载与预处理数据...")
    
    df = pd.read_csv(coord_csv)
    df.columns = df.columns.str.strip().str.lower()
    
    df2 = pd.read_csv(time_csv, encoding="gb2312")
    df2.columns = df2.columns.str.strip()
    
    if not {"日期", "时间"}.issubset(df2.columns):
        raise ValueError("源数据缺少 '日期' 或 '时间' 列。")
        
    dt = pd.to_datetime(df2["日期"].astype(str).str.strip() + " " + df2["时间"].astype(str).str.strip(), errors="coerce")
    
    df["_datetime"] = dt
    df = df.dropna(subset=["_datetime"]).copy()
    
    # 提取 t_point 距离特征并归一化
    t_dist_df = pd.read_csv(t_dist_csv)
    t_dist_col = "distance_to_nearest_line" if "distance_to_nearest_line" in t_dist_df.columns else "distance_to_plane"
    df["distance_t"] = t_dist_df[t_dist_col]
    dt_min, dt_max = df["distance_t"].min(), df["distance_t"].max()
    df["dist_t_norm"] = (df["distance_t"] - dt_min) / (dt_max - dt_min + 1e-8)
    
    # 提取 xyz_point 距离特征并归一化
    xyz_dist_df = pd.read_csv(xyz_dist_csv)
    xyz_dist_col = "distance_to_plane" if "distance_to_plane" in xyz_dist_df.columns else "distance_to_nearest_line"
    df["distance_xyz"] = xyz_dist_df[xyz_dist_col]
    dxyz_min, dxyz_max = df["distance_xyz"].min(), df["distance_xyz"].max()
    df["dist_xyz_norm"] = (df["distance_xyz"] - dxyz_min) / (dxyz_max - dxyz_min + 1e-8)
    
    if "e" in df.columns:
        df["e"] = pd.to_numeric(df["e"], errors="coerce")
        df["e_log"] = np.log10(1.0 + df["e"].clip(lower=0.0))
    else:
        df["e_log"] = 1.0 
    
    # 新增: 记录各维度的真实物理跨度，保存在 df 的 attrs 中供后续动态换算使用
    spatial_ranges = {}
    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise KeyError(f"主数据表中找不到需要的列: '{col}'")
        c_min, c_max = df[col].min(), df[col].max()
        spatial_ranges[col] = c_max - c_min + 1e-8
        df[f"{col}_norm"] = (df[col] - c_min) / (spatial_ranges[col])
        
    df.attrs["spatial_ranges"] = spatial_ranges
    return df

def generate_windows(df: pd.DataFrame, window_hours: float = 24.0, slide_hours: float = 24.0):
    df = df.sort_values("_datetime").reset_index(drop=True)
    start_time = df["_datetime"].min()
    end_time = df["_datetime"].max()
    
    windows = []
    current_time = start_time
    
    while current_time <= end_time:
        win_end = current_time + timedelta(hours=window_hours)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        window_df = df[mask]
        
        # 将命名格式修改为 YYYY-MM-DD_HH，更好区分滑动窗口的起始时间
        label = current_time.strftime("%Y-%m-%d_%H")
        windows.append((window_df, label))
        current_time += timedelta(hours=slide_hours)
        
    return windows

# ==========================================
# 模块3: 主流程控制与图像保存
# ==========================================

def save_image(img_array: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)

def save_heatmap(img_array: np.ndarray, path: str, cmap_name: str = "magma", use_gamma: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_clean = np.clip(img_array, 0, 1)
    
    if use_gamma:
        gamma = 0.3 
        img_clean = np.power(img_clean, gamma)
    
    cmap = plt.get_cmap(cmap_name)
    color_img = cmap(img_clean)
    img_uint8 = (color_img[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    
def evaluate_and_plot_multigauss(raw_target: np.ndarray, u_vals: np.ndarray, v_vals: np.ndarray, e_true: np.ndarray, e_log: np.ndarray, label: str, out_dir: str, proj_name: str, grid_size: int = 256, sigma: float = 16.0):
    import numpy.linalg as la

    valid = e_log > 0
    u, v, et, el = u_vals[valid], v_vals[valid], e_true[valid], e_log[valid]
    N = len(el)
    if N == 0: return None, None, None, None

    sort_idx = np.argsort(-et)
    et_sorted, el_sorted = et[sort_idx], el[sort_idx]
    
    cumsum_e = np.cumsum(et_sorted)
    top_k = np.searchsorted(cumsum_e, 0.90 * cumsum_e[-1]) + 1
    top_k = min(top_k, N)
    if top_k < 1: return None, None, None, None

    u_pix_int = np.clip(np.round(u * (grid_size - 1)).astype(int), 0, grid_size - 1)
    v_pix_int = np.clip(np.round(v * (grid_size - 1)).astype(int), 0, grid_size - 1)
    naive_est = raw_target[v_pix_int, u_pix_int]
    naive_est_sorted = naive_est[sort_idx]
    naive_errors = np.abs(naive_est_sorted[:top_k] - el_sorted[:top_k])

    y_grid, x_grid = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    y_flat, x_grid_flat = y_grid.flatten(), x_grid.flatten()
    u_pix, v_pix = u * (grid_size - 1), v * (grid_size - 1)
    
    M = np.zeros((grid_size * grid_size, N + 1), dtype=np.float32)
    for i in range(N):
        dist_sq = (x_grid_flat - u_pix[i])**2 + (y_flat - v_pix[i])**2
        M[:, i] = np.exp(-dist_sq / (2 * sigma**2))
    M[:, -1] = 1.0 
    
    Y = raw_target.flatten()
    
    A_est, _, _, _ = la.lstsq(M, Y, rcond=None)
    A_est_sorted = A_est[:N][sort_idx] 
    mg_errors = np.abs(A_est_sorted[:top_k] - el_sorted[:top_k])
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, top_k + 1), naive_errors, marker='x', markersize=5, linestyle='--', color='#7f7f7f', label=f'Naive Diff (Mean: {naive_errors.mean():.4f})')
    plt.plot(range(1, top_k + 1), mg_errors, marker='o', markersize=4, linestyle='-', color='#d62728', linewidth=1.5, label=f'Multi-Gauss Diff (Mean: {mg_errors.mean():.4f})')
    
    plt.title(f"Estimation Error Comparison (Top 90% Energy) - {label}\nMulti-Gauss Avg Err: {mg_errors.mean():.4f} vs Naive Avg Err: {naive_errors.mean():.4f}")
    plt.xlabel("Microseismic Event Rank")
    plt.ylabel("Absolute Est. Error in LogD Domain")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plot_dir = os.path.join(out_dir, "fidelity_eval_multigauss", proj_name)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"{label}.png"), bbox_inches='tight', dpi=150)
    plt.close()

    return mg_errors.max(), mg_errors.mean(), naive_errors.max(), naive_errors.mean()


def main():
    coord_csv = r"data/7303_xyzet_rotated.csv"
    time_csv = r"data/7303_all.csv"
    
    # 距离计算结果文件路径
    t_dist_csv = r"script_calculate_distance/result/t_point_distances.csv"
    xyz_dist_csv = r"script_calculate_distance/result/xyz_point_distances.csv"
    
    # 修改输出路径结构
    out_energy = r"input_st2/e_data"
    out_t_cov = r"input_st2/t_cov1"
    out_xyz_cov = r"input_st2/xyz_cov2"
    
    grid_size = 256
    
    df = load_and_preprocess_data(coord_csv, time_csv, t_dist_csv, xyz_dist_csv)
    spatial_ranges = df.attrs["spatial_ranges"]
    
    # 直接使用滑动窗口切分数据（例如：窗口24小时，步长12小时，即可天然包含中间状态）
    window_hours = 24.0
    slide_hours = 12.0 #24.0 #
    windows = generate_windows(df, window_hours=window_hours, slide_hours=slide_hours)
    
    print(f"按窗口大小 {window_hours}h, 步长 {slide_hours}h 切分，共生成 {len(windows)} 个滑动窗口。\n")
    
    global_valid_pixels = []
    
    # =======================================================
    # 新增：定义具有物理意义的各向异性带宽参数与截断物理时长
    # =======================================================
    PHYSICAL_SIGMA_T_HOURS = 0.48#1.5   # 时间长尾
    
    # 空间上的各向异性（单位：米）
    # X轴（如走向）：破裂易沿煤层延伸，范围较广
    PHYSICAL_SIGMA_S_METERS_X = 68.18#80.0 
    # Y轴（如倾向）：受工作面宽度限制或保护煤柱影响，一般与X接近或稍小
    PHYSICAL_SIGMA_S_METERS_Y = 30.11#60.0 
    # Z轴（垂向）：穿层破裂受坚硬顶底板阻隔，衰减最快，通常为平面的一半或更小
    PHYSICAL_SIGMA_S_METERS_Z = 7.56#30.0 
    
    # 将各维度的物理带宽整合为字典供循环调用
    PHYSICAL_SIGMA_S_METERS_MAP = {
        "x": PHYSICAL_SIGMA_S_METERS_X,
        "y": PHYSICAL_SIGMA_S_METERS_Y,
        "z": PHYSICAL_SIGMA_S_METERS_Z
    }
    
    # 截断时长：...
    HISTORY_TRUNCATE_HOURS = PHYSICAL_SIGMA_T_HOURS * 3.0
        
    history_dfs = [] # 用于保存历史微震事件
    
    for win_df, label in windows:
        if len(win_df) == 0:
            print(f"跳过空窗口: {label}")
            continue
            
        # 【修改点1】：提取严格的窗口物理边界点 (以时间戳秒数计算)
        # label 格式形如 "2023-07-07_00"
        win_start_dt = pd.to_datetime(label, format="%Y-%m-%d_%H")
        win_end_dt = win_start_dt + timedelta(hours=window_hours)
        
        # 将 datetime 转为绝对时间戳（秒）作为坐标基准
        win_t_start = win_start_dt.timestamp()
        win_t_end = win_end_dt.timestamp()
        
        # 将当前窗口加入历史列表
        history_dfs.append(win_df)
        
        # 清理过于久远的历史事件
        # 只保留 t (已经是归一化前的真实尺度即可，前提是 df["t"] 是绝对时间或者相对时间，
        # 如果 df["t"] 是基于第一天的相对秒数，我们需要统一基准：
        # 这里假定生成窗口时 df 里的 datetime 可以作为绝对基准)
        
        # 修正：直接用 _datetime 计算秒数进行截断和映射
        truncate_threshold_dt = win_start_dt - timedelta(hours=HISTORY_TRUNCATE_HOURS)
        valid_history = []
        for h_df in history_dfs:
            h_df_valid = h_df[h_df["_datetime"] >= truncate_threshold_dt]
            if not h_df_valid.empty:
                valid_history.append(h_df_valid)
        history_dfs = valid_history
        
        combined_df = pd.concat(history_dfs)
        
        print(f"正在处理窗口: {label} | 包含 {len(win_df)} 个源点 (结合近期历史共 {len(combined_df)} 个点参与渲染)")
        
        # 提取全量渲染点的空间和能量特征
        x = combined_df["x_norm"].values
        y = combined_df["y_norm"].values
        z = combined_df["z_norm"].values
        e_log = combined_df["e_log"].values
        e_true = pd.to_numeric(combined_df["e"], errors="coerce").values
        dist_t_val = combined_df["dist_t_norm"].values
        dist_xyz_val = combined_df["dist_xyz_norm"].values
        
        # 【修改点2】：将事件的时间，严格映射到 [窗口起点, 窗口终点] 的 0-1 空间
        t_raw_all = combined_df["_datetime"].apply(lambda x: x.timestamp()).values
        t = (t_raw_all - win_t_start) / ((win_t_end - win_t_start) + 1e-8)
        
        win_duration_hours = window_hours
        
        # 计算当前物理带宽对应的像素带宽
        sigma_u_pixels = (PHYSICAL_SIGMA_T_HOURS / win_duration_hours) * grid_size
        
        projections = [("tx", t, x, "x"), ("ty", t, y, "y"), ("tz", t, z, "z")]
        
        for proj_name, u_vals, v_vals, space_col in projections:
            # 【修改】：根据当前的投影轴动态获取其专属的物理带宽，并计算其像素带宽
            current_physical_sigma_s = PHYSICAL_SIGMA_S_METERS_MAP[space_col]
            sigma_v_pixels = (current_physical_sigma_s / spatial_ranges[space_col]) * grid_size
            
            # ================= 1. 能量场映射 =================
            raw_target = render_gaussian_aniso(u_vals, v_vals, e_log, grid_size=grid_size, sigma_u=sigma_u_pixels, sigma_v=sigma_v_pixels)
            
            if raw_target.max() > raw_target.min():
                raw_vis = (raw_target - raw_target.min()) / (raw_target.max() - raw_target.min())
            else:
                raw_vis = np.zeros_like(raw_target)
            
            os.makedirs(os.path.join(out_energy, "raw_interp", proj_name), exist_ok=True)
            raw_save_path = os.path.join(out_energy, "raw_interp", proj_name, f"{label}.png")
            save_image(raw_vis, raw_save_path)
            save_heatmap(raw_vis, raw_save_path.replace("raw_interp", "raw_interp_vis"), use_gamma=True)
            
            npy_save_dir = os.path.join(out_energy, "matrix_npy", proj_name)
            os.makedirs(npy_save_dir, exist_ok=True)
            np.save(os.path.join(npy_save_dir, f"{label}.npy"), raw_target)
            
            # 多高斯误差评估 (只评估当前窗口的真实误差，为了避免历史长尾点干扰，需截取当前窗口点)
            # 不过这里为了简单传入组合后的真实能量也可以，只是会有更多点参与渲染和评价
            # evaluate_and_plot_multigauss(...)
            
            cdf_mapped = apply_quantile_mapping(raw_target)
            valid_data = cdf_mapped[cdf_mapped > 0].flatten()
            if proj_name == 'tx': 
                global_valid_pixels.append(valid_data)
                
            os.makedirs(os.path.join(out_energy, "cdf_mapped", proj_name), exist_ok=True)
            cdf_save_path = os.path.join(out_energy, "cdf_mapped", proj_name, f"{label}.png")
            save_image(cdf_mapped, cdf_save_path)
            save_heatmap(cdf_mapped, cdf_save_path.replace("cdf_mapped", "cdf_mapped_vis"), cmap_name="turbo", use_gamma=False)

            # ================= 2. T 距离协变量场映射 =================
            proximity_t_val = 1.0 - dist_t_val
            dist_t_target = render_gaussian_aniso(u_vals, v_vals, proximity_t_val, grid_size=grid_size, sigma_u=sigma_u_pixels, sigma_v=sigma_v_pixels)
            
            t_npy_dir = os.path.join(out_t_cov, "matrix_npy", proj_name)
            os.makedirs(t_npy_dir, exist_ok=True)
            np.save(os.path.join(t_npy_dir, f"{label}.npy"), dist_t_target)
            
            if dist_t_target.max() > dist_t_target.min():
                dist_t_vis = (dist_t_target - dist_t_target.min()) / (dist_t_target.max() - dist_t_target.min())
            else:
                dist_t_vis = np.zeros_like(dist_t_target)
            t_vis_dir = os.path.join(out_t_cov, "cov_distance_vis", proj_name)
            os.makedirs(t_vis_dir, exist_ok=True)
            save_heatmap(dist_t_vis, os.path.join(t_vis_dir, f"{label}.png"), cmap_name="viridis", use_gamma=False)

            # ================= 3. XYZ 距离协变量场映射 =================
            proximity_xyz_val = 1.0 - dist_xyz_val
            dist_xyz_target = render_gaussian_aniso(u_vals, v_vals, proximity_xyz_val, grid_size=grid_size, sigma_u=sigma_u_pixels, sigma_v=sigma_v_pixels)
            
            xyz_npy_dir = os.path.join(out_xyz_cov, "matrix_npy", proj_name)
            os.makedirs(xyz_npy_dir, exist_ok=True)
            np.save(os.path.join(xyz_npy_dir, f"{label}.npy"), dist_xyz_target)
            
            if dist_xyz_target.max() > dist_xyz_target.min():
                dist_xyz_vis = (dist_xyz_target - dist_xyz_target.min()) / (dist_xyz_target.max() - dist_xyz_target.min())
            else:
                dist_xyz_vis = np.zeros_like(dist_xyz_target)
            xyz_vis_dir = os.path.join(out_xyz_cov, "cov_distance_vis", proj_name)
            os.makedirs(xyz_vis_dir, exist_ok=True)
            save_heatmap(dist_xyz_vis, os.path.join(xyz_vis_dir, f"{label}.png"), cmap_name="viridis", use_gamma=False)

    print("\n✅ 所有处理完成！结果均已分发到各自对应的 input 目录下。")

    if windows and global_valid_pixels:
        plt.figure(figsize=(14, 7))
        all_data = np.concatenate(global_valid_pixels)
        
        hist_densities, bin_edges = np.histogram(all_data, bins=50, range=(0, 1), density=True)
        mae_density = np.mean(np.abs(hist_densities - 1.0))
        uniformity_score = max(0.0, (1.0 - mae_density) * 100)
        
        print(f"\n📈 全局 CDF 映射质量评估:")
        print(f" - 密度平均绝对误差 (MAE): {mae_density:.4f}")
        print(f" - 数据分布均匀度得分: {uniformity_score:.2f}% (100% 为极端完美状态)")
        
        plt.figure(figsize=(8, 5))
        plt.hist(all_data, bins=50, range=(0,1), density=True, color='teal', alpha=0.7)
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Uniform (y=1.0)')
        
        plt.title(f"Global CDF Uniform Distribution\nUniformity Score: {uniformity_score:.2f}%")
        plt.xlabel("Mapped Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        
        hist_save_path = os.path.join(out_energy, "global_cdf_histogram.png")
        plt.savefig(hist_save_path, dpi=300)
        print(f"📊 全局数据分布直方图已保存至: {hist_save_path}\n")


if __name__ == "__main__":
    if HAS_CUPY:
        print("🚀 已检测到 CuPy，高斯渲染将通过 GPU 加速执行。")
    main()