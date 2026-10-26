import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy.linalg as la

# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
    'font.size': 14,               
    'axes.labelsize': 15,          
    'xtick.labelsize': 13,         
    'ytick.labelsize': 13,
    'svg.fonttype': 'path',
})

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = np

def render_gaussian_aniso(u, v, e, grid_size=256, sigma_u=8.0, sigma_v=16.0):
    xp = cp if HAS_CUPY else np
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    e_val = xp.asarray(e, dtype=xp.float32)
    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)
    for i in range(len(u_pix)):
        if e_val[i] <= 0: continue
        dist_sq_u = (x_grid - u_pix[i])**2 / (2 * sigma_u**2)
        dist_sq_v = (y_grid - v_pix[i])**2 / (2 * sigma_v**2)
        canvas += e_val[i] * xp.exp(-(dist_sq_u + dist_sq_v))
    return cp.asnumpy(canvas) if HAS_CUPY else canvas

def load_and_preprocess_data(coord_csv, time_csv):
    df = pd.read_csv(coord_csv)
    df.columns = df.columns.str.strip().str.lower()
    df2 = pd.read_csv(time_csv, encoding="gb2312")
    df2.columns = df2.columns.str.strip()
    dt = pd.to_datetime(df2["日期"].astype(str).str.strip() + " " + df2["时间"].astype(str).str.strip(), errors="coerce")
    df["_datetime"] = dt
    df = df.dropna(subset=["_datetime"]).copy()
    if "e" in df.columns:
        df["e"] = pd.to_numeric(df["e"], errors="coerce")
        df["e_log"] = np.log10(1.0 + df["e"].clip(lower=0.0))
    else:
        df["e_log"] = 1.0 
    spatial_ranges = {}
    for col in ["x", "y", "z"]:
        c_min, c_max = df[col].min(), df[col].max()
        spatial_ranges[col] = c_max - c_min + 1e-8
        df[f"{col}_norm"] = (df[col] - c_min) / (spatial_ranges[col])
    df.attrs["spatial_ranges"] = spatial_ranges
    return df

def generate_windows(df, window_hours=24.0, slide_hours=24.0):
    df = df.sort_values("_datetime").reset_index(drop=True)
    start_time = df["_datetime"].min()
    end_time = df["_datetime"].max()
    windows = []
    current_time = start_time
    while current_time <= end_time:
        win_end = current_time + timedelta(hours=window_hours)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        window_df = df[mask]
        label = current_time.strftime("%Y-%m-%d_%H")
        windows.append((window_df, label))
        current_time += timedelta(hours=slide_hours)
    return windows

def evaluate_multigauss_errors(raw_target, u_vals, v_vals, e_true, e_log, grid_size=256, sigma_u=16.0, sigma_v=16.0):
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
        # 【关键修改】：支持各向异性带宽的误差评估
        dist_sq_u = (x_grid_flat - u_pix[i])**2 / (2 * sigma_u**2)
        dist_sq_v = (y_flat - v_pix[i])**2 / (2 * sigma_v**2)
        M[:, i] = np.exp(-(dist_sq_u + dist_sq_v))
    M[:, -1] = 1.0 
    
    Y = raw_target.flatten()
    
    A_est, _, _, _ = la.lstsq(M, Y, rcond=None)
    A_est_sorted = A_est[:N][sort_idx] 
    mg_errors = np.abs(A_est_sorted[:top_k] - el_sorted[:top_k])
    
    return mg_errors.max(), mg_errors.mean(), naive_errors.max(), naive_errors.mean()

def main():
    coord_csv = r"data/7303_xyzet_rotated.csv"
    time_csv = r"data/7303_all.csv"
    grid_size = 256
    
    df = load_and_preprocess_data(coord_csv, time_csv)
    spatial_ranges = df.attrs["spatial_ranges"]
    
    window_hours = 24.0
    slide_hours = 12.0
    windows = generate_windows(df, window_hours=window_hours, slide_hours=slide_hours)
    
    print(f"按窗口大小 {window_hours}h, 步长 {slide_hours}h 切分，共生成 {len(windows)} 个滑动窗口。\n")
    
    projs = ['tx', 'ty', 'tz', 'xy', 'xz', 'yz']
    global_mg_max_err = {p: [] for p in projs}
    global_mg_mean_err = {p: [] for p in projs}
    global_nv_max_err = {p: [] for p in projs}
    global_nv_mean_err = {p: [] for p in projs}
    frame_labels = []
    
    PHYSICAL_SIGMA_T_HOURS = 12
    PHYSICAL_SIGMA_S_METERS_X = 68.18
    PHYSICAL_SIGMA_S_METERS_Y = 30.11
    PHYSICAL_SIGMA_S_METERS_Z = 7.56
    
    PHYSICAL_SIGMA_S_METERS_MAP = {
        "x": PHYSICAL_SIGMA_S_METERS_X,
        "y": PHYSICAL_SIGMA_S_METERS_Y,
        "z": PHYSICAL_SIGMA_S_METERS_Z
    }
    
    HISTORY_TRUNCATE_HOURS = PHYSICAL_SIGMA_T_HOURS * 3.0
    history_dfs = []
    
    for win_df, label in windows:
        win_start_dt = pd.to_datetime(label, format="%Y-%m-%d_%H")
        win_end_dt = win_start_dt + timedelta(hours=window_hours)
        win_t_start = win_start_dt.timestamp()
        win_t_end = win_end_dt.timestamp()
        
        history_dfs.append(win_df)
        
        truncate_threshold_dt = win_start_dt - timedelta(hours=HISTORY_TRUNCATE_HOURS)
        valid_history = []
        for h_df in history_dfs:
            h_df_valid = h_df[h_df["_datetime"] >= truncate_threshold_dt]
            if not h_df_valid.empty:
                valid_history.append(h_df_valid)
        history_dfs = valid_history
        
        combined_df = pd.concat(history_dfs)
        print(f"正在处理窗口: {label} | 包含 {len(win_df)} 个源点")
        
        x = combined_df["x_norm"].values
        y = combined_df["y_norm"].values
        z = combined_df["z_norm"].values
        e_log = combined_df["e_log"].values
        e_true = pd.to_numeric(combined_df["e"], errors="coerce").values
        
        t_raw_all = combined_df["_datetime"].apply(lambda x: x.timestamp()).values
        t = (t_raw_all - win_t_start) / ((win_t_end - win_t_start) + 1e-8)
        
        sigma_t_pixels = (PHYSICAL_SIGMA_T_HOURS / window_hours) * grid_size
        
        delta_t_hours = (win_t_end - t_raw_all) / 3600.0
        delta_t_hours = np.maximum(delta_t_hours, 0)
        time_weight = np.exp(- (delta_t_hours**2) / (2 * PHYSICAL_SIGMA_T_HOURS**2))
        
        e_render_spatial = e_log * time_weight
        e_true_spatial = e_true * time_weight
        
        projections = [
            ("tx", t, x, "t", "x", e_log, e_true),
            ("ty", t, y, "t", "y", e_log, e_true),
            ("tz", t, z, "t", "z", e_log, e_true),
            ("xy", x, y, "x", "y", e_render_spatial, e_true_spatial),
            ("xz", x, z, "x", "z", e_render_spatial, e_true_spatial),
            ("yz", y, z, "y", "z", e_render_spatial, e_true_spatial)
        ]
        
        for proj_name, u_vals, v_vals, col_u, col_v, e_vals, et_vals in projections:
            if col_u == "t":
                sigma_u_pixels = sigma_t_pixels
            else:
                sigma_u_pixels = (PHYSICAL_SIGMA_S_METERS_MAP[col_u] / spatial_ranges[col_u]) * grid_size
                
            sigma_v_pixels = (PHYSICAL_SIGMA_S_METERS_MAP[col_v] / spatial_ranges[col_v]) * grid_size
            
            raw_target = render_gaussian_aniso(u_vals, v_vals, e_vals, grid_size=grid_size, sigma_u=sigma_u_pixels, sigma_v=sigma_v_pixels)
            
            mg_max, mg_mean, nv_max, nv_mean = evaluate_multigauss_errors(
                raw_target, u_vals, v_vals, et_vals, e_vals, grid_size, sigma_u_pixels, sigma_v_pixels
            )
            
            if mg_max is not None:
                global_mg_max_err[proj_name].append(mg_max)
                global_mg_mean_err[proj_name].append(mg_mean)
                global_nv_max_err[proj_name].append(nv_max)
                global_nv_mean_err[proj_name].append(nv_mean)
            else:
                global_mg_max_err[proj_name].append(0)
                global_mg_mean_err[proj_name].append(0)
                global_nv_max_err[proj_name].append(0)
                global_nv_mean_err[proj_name].append(0)
                
        frame_labels.append(label)

    print("\n✅ 误差计算完成，开始绘制趋势图...")

    if frame_labels:
        trend_out_dir = os.path.join("another", "chazhi_txyz_ST-KDE")
        os.makedirs(trend_out_dir, exist_ok=True)
        
        x_ticks = range(len(frame_labels))
        colors = {'tx': '#1f77b4', 'ty': '#ff7f0e', 'tz': '#2ca02c', 'xy': '#d62728', 'xz': '#9467bd', 'yz': '#8c564b'}
        
        # 图1: 全局平均误差趋势对比
        fig, ax = plt.subplots(figsize=(14, 7))
        for proj in projs:
            ax.plot(x_ticks, global_nv_mean_err[proj], color=colors[proj], linestyle=':', alpha=0.6, label=f'朴素插值均值误差 ({proj.upper()})')
            ax.plot(x_ticks, global_mg_mean_err[proj], color=colors[proj], marker='.', linewidth=2, label=f'多高斯插值均值误差 ({proj.upper()})')
        
        ax.set_xlabel('时间窗口序列')
        ax.set_ylabel('绝对误差 (对数域)')
        ax.set_xticks(x_ticks[::max(1, len(x_ticks)//10)])
        ax.set_xticklabels([frame_labels[i] for i in x_ticks[::max(1, len(x_ticks)//10)]], rotation=45)
        ax.grid(True, alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        fig.canvas.draw()
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label_obj in tick_labels:
            label_obj.set_fontname('Times New Roman')
            
        plt.tight_layout()
        plt.savefig(os.path.join(trend_out_dir, "global_trend_mean_error.png"), dpi=300)
        plt.savefig(os.path.join(trend_out_dir, "global_trend_mean_error.svg"), format='svg')
        plt.close()

        # 图2: 全局最大误差趋势对比
        fig, ax = plt.subplots(figsize=(14, 7))
        for proj in projs:
            ax.plot(x_ticks, global_nv_max_err[proj], color=colors[proj], linestyle=':', alpha=0.6, label=f'朴素插值最大误差 ({proj.upper()})')
            ax.plot(x_ticks, global_mg_max_err[proj], color=colors[proj], marker='.', linewidth=2, label=f'多高斯插值最大误差 ({proj.upper()})')
        
        ax.set_xlabel('时间窗口序列')
        ax.set_ylabel('绝对误差 (对数域)')
        ax.set_xticks(x_ticks[::max(1, len(x_ticks)//10)])
        ax.set_xticklabels([frame_labels[i] for i in x_ticks[::max(1, len(x_ticks)//10)]], rotation=45)
        ax.grid(True, alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        formatter = plt.ScalarFormatter(useOffset=False)
        formatter.set_useMathText(True)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        fig.canvas.draw()
        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label_obj in tick_labels:
            label_obj.set_fontname('Times New Roman')
            
        plt.tight_layout()
        plt.savefig(os.path.join(trend_out_dir, "global_trend_max_error.png"), dpi=300)
        plt.savefig(os.path.join(trend_out_dir, "global_trend_max_error.svg"), format='svg')
        plt.close()
        
        print(f"\n📈 全局拟合极值与均值误差折线图已保存至: {trend_out_dir}")

if __name__ == "__main__":
    if HAS_CUPY:
        print("🚀 已检测到 CuPy，高斯渲染将通过 GPU 加速执行。")
    main()
