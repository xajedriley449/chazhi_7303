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

def render_gaussian_sum(u: np.ndarray, v: np.ndarray, e: np.ndarray, grid_size: int = 256, sigma: float = 12.0):
    """
    步骤1：高斯累加插值 (Log-Gaussian KDE-Sum)
    """
    xp = cp if HAS_CUPY else np
    
    # 将坐标映射到像素索引范围
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    e_val = xp.asarray(e, dtype=xp.float32)
    
    # 废除正方形包围盒！为了兼容后续的 Gamma 提亮，必须保证尾部平滑过渡至无限远。
    # 256x256 分辨率极小，直接全图计算不仅依然保持毫秒级极速，而且保证了纯粹的浑圆无台阶。
    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)
    
    for i in range(len(u_pix)):
        if e_val[i] <= 0: continue
        # 全图计算距离平方
        dist_sq = (x_grid - u_pix[i])**2 + (y_grid - v_pix[i])**2
        # 全图叠加高斯分布
        canvas += e_val[i] * xp.exp(-dist_sq / (2 * sigma**2))

    return cp.asnumpy(canvas) if HAS_CUPY else canvas

def render_gaussian_max(u: np.ndarray, v: np.ndarray, e: np.ndarray, grid_size: int = 256, sigma: float = 16.0):
    """
    采用【高斯核 + 最大值合成】策略。
    完美保证核心高能点的绝对峰值不失真，同时以平滑的高斯尾巴避免产生影响 CNN 提取特征的硬边缘。
    """
    xp = cp if HAS_CUPY else np
    
    u_pix = xp.asarray(u * (grid_size - 1), dtype=xp.float32)
    v_pix = xp.asarray(v * (grid_size - 1), dtype=xp.float32)
    e_val = xp.asarray(e, dtype=xp.float32)
    
    y_grid, x_grid = xp.mgrid[0:grid_size, 0:grid_size].astype(xp.float32)
    canvas = xp.zeros((grid_size, grid_size), dtype=xp.float32)
    
    for i in range(len(u_pix)):
        if e_val[i] <= 0: continue
        
        # 计算全图距离的平方
        dist_sq = (x_grid - u_pix[i])**2 + (y_grid - v_pix[i])**2
        
        # 生成当前单个点的高斯面 (峰值严格等于 e_val[i])
        single_gaussian = e_val[i] * xp.exp(-dist_sq / (2 * sigma**2))
        
        # 【关键修改】使用最大值覆盖，避免抱团虚高，确保峰值绝对保真
        canvas = xp.maximum(canvas, single_gaussian)

    return cp.asnumpy(canvas) if HAS_CUPY else canvas

def apply_quantile_mapping(image_matrix: np.ndarray, background_threshold: float = 1e-5):
    """
    步骤2：相对分位数映射 (Quantile CDF Mapping)
    将无明显边界波动的能量值映射为相对 [0, 1] 的风险概率。
    """
    H, W = image_matrix.shape
    flat_img = image_matrix.flatten().reshape(-1, 1)
    
    # 构建有效区域掩码（忽略背景死点，避免干扰分布）
    valid_mask = flat_img > background_threshold
    valid_pixels = flat_img[valid_mask].reshape(-1, 1)
    
    # 极端情况：如果整张图基本没有有效震动，直接返回全零
    if len(valid_pixels) < 10:
        return np.zeros_like(image_matrix)
        
    n_quantiles = min(1000, len(valid_pixels))
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    
    # 对有效能量点进行相对排名映射
    mapped_valid = qt.fit_transform(valid_pixels)
    
    # 将排完名的 [0, 1] 值填回空白画布
    mapped_img_flat = np.zeros_like(flat_img, dtype=np.float32)
    mapped_img_flat[valid_mask] = mapped_valid.flatten()
    
    return mapped_img_flat.reshape(H, W)


# ==========================================
# 模块2: 数据加载与时序窗口滑动
# ==========================================

def load_and_preprocess_data(coord_csv: str, time_csv: str):
    print("正在加载与预处理数据...")
    
    # 1. 加载旋转后的坐标与能量数据 (包含 x, y, z, e, t)
    df = pd.read_csv(coord_csv)
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. 加载带有原始日期时间的汇总文件
    df2 = pd.read_csv(time_csv, encoding="gb2312")
    df2.columns = df2.columns.str.strip()
    
    # 3. 提取并严格按行对齐时间 (与 chazhi.py 逻辑一致)
    if not {"日期", "时间"}.issubset(df2.columns):
        raise ValueError("源数据缺少 '日期' 或 '时间' 列。")
        
    dt = pd.to_datetime(df2["日期"].astype(str).str.strip() + " " + df2["时间"].astype(str).str.strip(), errors="coerce")
    
    # 将时间赋给主表并剔除无效行
    df["_datetime"] = dt
    df = df.dropna(subset=["_datetime"]).copy()
    
    # 4. 对能量进行 Log 降维，暴削极端长尾
    if "e" in df.columns:
        df["e"] = pd.to_numeric(df["e"], errors="coerce")
        df["e_log"] = np.log10(1.0 + df["e"].clip(lower=0.0))
    else:
        df["e_log"] = 1.0 # 容错
    
    # 5. 将 X, Y, Z 进行全局的统一 Min-Max 归一化到 [0, 1] (注意：踢出 t，t 需随窗口局部归一化)
    for col in ["x", "y", "z"]:
        if col not in df.columns:
            raise KeyError(f"主数据表中找不到需要的列: '{col}'")
        c_min, c_max = df[col].min(), df[col].max()
        df[f"{col}_norm"] = (df[col] - c_min) / (c_max - c_min + 1e-8)
        
    return df

def generate_windows(df: pd.DataFrame, window_hours: float = 24.0, slide_hours: float = 24.0):
    """基于固定的物理时间滑动切分数据帧"""
    df = df.sort_values("_datetime").reset_index(drop=True)
    start_time = df["_datetime"].min()
    end_time = df["_datetime"].max()
    
    windows = []
    current_time = start_time
    
    while current_time <= end_time:
        win_end = current_time + timedelta(hours=window_hours)
        mask = (df["_datetime"] >= current_time) & (df["_datetime"] < win_end)
        window_df = df[mask]
        
        label = current_time.strftime("%Y%m%d_%H%M")
        windows.append((window_df, label))
        current_time += timedelta(hours=slide_hours)
        
    return windows

# ==========================================
# 模块3: 主流程控制与图像保存
# ==========================================

def save_image(img_array: np.ndarray, path: str):
    """将 [0,1] 的浮点数组保存为 8-bit png (模型训练专用)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)

def save_heatmap(img_array: np.ndarray, path: str, cmap_name: str = "magma", use_gamma: bool = False):
    """使用伪彩映射保存为RGB图，专供人类观察"""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    img_clean = np.clip(img_array, 0, 1)
    
    # 只有 raw 这种拖尾极度严重的数据才需要通过 gamma 强行提亮暗部
    if use_gamma:
        gamma = 0.3 
        img_clean = np.power(img_clean, gamma)
    
    cmap = plt.get_cmap(cmap_name)
    color_img = cmap(img_clean)
    img_uint8 = (color_img[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    
# 【新增】多高斯联合解算与Top90%能量误差折线图评估
def evaluate_and_plot_multigauss(raw_target: np.ndarray, u_vals: np.ndarray, v_vals: np.ndarray, e_true: np.ndarray, e_log: np.ndarray, label: str, out_dir: str, proj_name: str, grid_size: int = 256, sigma: float = 16.0):
    import matplotlib.pyplot as plt
    import numpy.linalg as la

    # 1. 过滤死点
    valid = e_log > 0
    u, v, et, el = u_vals[valid], v_vals[valid], e_true[valid], e_log[valid]
    N = len(el)
    if N == 0: return None, None, None, None

    # 2. 按原始真实能量降序，并截取累积百分之90的微震主要事件
    sort_idx = np.argsort(-et)
    et_sorted, el_sorted = et[sort_idx], el[sort_idx]
    
    cumsum_e = np.cumsum(et_sorted)
    top_k = np.searchsorted(cumsum_e, 0.90 * cumsum_e[-1]) + 1
    top_k = min(top_k, N)
    if top_k < 1: return None, None, None, None

    # --- A. 获取【朴素估计】 (直接读取图上对应坐标的像素值，叠加会导致严重偏大) ---
    u_pix_int = np.clip(np.round(u * (grid_size - 1)).astype(int), 0, grid_size - 1)
    v_pix_int = np.clip(np.round(v * (grid_size - 1)).astype(int), 0, grid_size - 1)
    naive_est = raw_target[v_pix_int, u_pix_int]
    naive_est_sorted = naive_est[sort_idx]
    naive_errors = np.abs(naive_est_sorted[:top_k] - el_sorted[:top_k])

    # --- B. 获取【多高斯联合解算估值】 (分离解耦) ---
    y_grid, x_grid = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
    y_flat, x_flat = y_grid.flatten(), x_grid.flatten()
    u_pix, v_pix = u * (grid_size - 1), v * (grid_size - 1)
    
    M = np.zeros((grid_size * grid_size, N + 1), dtype=np.float32)
    for i in range(N):
        dist_sq = (x_flat - u_pix[i])**2 + (y_flat - v_pix[i])**2
        M[:, i] = np.exp(-dist_sq / (2 * sigma**2))
    M[:, -1] = 1.0  # 全局基底项
    
    Y = raw_target.flatten()
    
    A_est, _, _, _ = la.lstsq(M, Y, rcond=None)
    A_est_sorted = A_est[:N][sort_idx] 
    mg_errors = np.abs(A_est_sorted[:top_k] - el_sorted[:top_k])
    
    # 画对比折线图并保存
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, top_k + 1), naive_errors, marker='x', markersize=5, linestyle='--', color='#7f7f7f', label=f'Naive Diff (Mean: {naive_errors.mean():.4f})')
    plt.plot(range(1, top_k + 1), mg_errors, marker='o', markersize=4, linestyle='-', color='#d62728', linewidth=1.5, label=f'Multi-Gauss Diff (Mean: {mg_errors.mean():.4f})')
    
    plt.title(f"Estimation Error Comparison (Top 90% Energy) - {label}\nMulti-Gauss Avg Err: {mg_errors.mean():.4f} vs Naive Avg Err: {naive_errors.mean():.4f}")
    plt.xlabel("Microseismic Event Rank (Sorted by actual large energy)")
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
    out_dir = r"output_cdf/daily/channel_0"
    grid_size = 256
    
    # 控制滑动窗口和插值密度
    window_h = 24.0
    slide_h = 12.0 #+ 6.0 # 每12小时滑动一次，叠加6小时重叠，形成更平滑的过渡帧
    
    df = load_and_preprocess_data(coord_csv, time_csv)
    windows = generate_windows(df, window_h, slide_h)
    print(f"共生成 {len(windows)} 个有效时间帧。准备开始渲染映射...\n")
    
    # 【新增】用于收集全局所有有效像素，验证整体分布
    global_valid_pixels = []
    
    # 【新增】用于收集跨时间帧的各个投影通道拟合误差
    global_mg_max_err = {'tx': [], 'ty': [], 'tz': []}
    global_mg_mean_err = {'tx': [], 'ty': [], 'tz': []}
    global_nv_max_err = {'tx': [], 'ty': [], 'tz': []}
    global_nv_mean_err = {'tx': [], 'ty': [], 'tz': []}
    frame_labels = []
    
    for i, (win_df, label) in enumerate(windows):
        if len(win_df) < 2:
            continue # 点数太少，跳过空帧
            
        print(f"处理第 [{i+1}/{len(windows)}] 帧: {label} | 包含 {len(win_df)} 个源点")
        frame_labels.append(label)
        
        # 提取需要的数据，对于 t 轴必须进行**窗口内部**的局部归一化
        x = win_df["x_norm"].values
        y = win_df["y_norm"].values
        z = win_df["z_norm"].values
        
        t_raw = pd.to_numeric(win_df["t"], errors="coerce").values
        if len(t_raw) > 1 and t_raw.max() > t_raw.min():
            t = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-8)
        else:
            t = np.zeros_like(t_raw)
            
        e_log = win_df["e_log"].values
        # 【新增】提取绝对真实能量，用于 Top90% 排序比较
        e_true = pd.to_numeric(win_df["e"], errors="coerce").values
        
        # 针对 TX, TY, TZ 三个投影分别：插值 + CDF 映射！
        projections = [
            ("tx", t, x),
            ("ty", t, y),
            ("tz", t, z)
        ]
        
        for proj_name, u_vals, v_vals in projections:
            # 1. 物理高斯累加 (生成连绵起伏的能量地形图)
            # 【重要】将 sigma 从之前的 3.0 放大到 15.0，让能量波及更广，形成融合的山脉而非孤零零的烟囱
            raw_target = render_gaussian_sum(u_vals, v_vals, e_log, grid_size=grid_size, sigma=16.0)
            # 1. 高斯最大值渲染 (替代原有的 Gaussian Sum)
            #raw_target = render_gaussian_max(u_vals, v_vals, e_log, grid_size=grid_size, sigma=16.0)

            # 【核心修改】接收两组误差
            mg_max, mg_mean, nv_max, nv_mean = evaluate_and_plot_multigauss(raw_target, u_vals, v_vals, e_true, e_log, label, out_dir, proj_name, grid_size, 16.0)
            if mg_max is not None:
                global_mg_max_err[proj_name].append(mg_max)
                global_mg_mean_err[proj_name].append(mg_mean)
                global_nv_max_err[proj_name].append(nv_max)
                global_nv_mean_err[proj_name].append(nv_mean)
            else:
                global_mg_max_err[proj_name].append(0); global_mg_mean_err[proj_name].append(0)
                global_nv_max_err[proj_name].append(0); global_nv_mean_err[proj_name].append(0)
            
            # 为了能够直观观察插值结果，将原始插值矩阵归一化到 [0, 1] 后保存
            if raw_target.max() > raw_target.min():
                raw_vis = (raw_target - raw_target.min()) / (raw_target.max() - raw_target.min())
            else:
                raw_vis = np.zeros_like(raw_target)
                
            raw_save_path = os.path.join(out_dir, "raw_interp", proj_name, f"{label}.png")
            save_image(raw_vis, raw_save_path)
            # Raw 因为绝对跨度大，需要 Gamma 提亮暗部才能看见
            save_heatmap(raw_vis, raw_save_path.replace("raw_interp", "raw_interp_vis"), use_gamma=True)
            
            # 【核心修复】保存无损的浮点矩阵，供神经网络模型直接输入！
            npy_save_dir = os.path.join(out_dir, "matrix_npy", proj_name)
            os.makedirs(npy_save_dir, exist_ok=True)
            np.save(os.path.join(npy_save_dir, f"{label}.npy"), raw_target)
            
            # 2. 连续极值分位数映射 (抹平绝对能量差异，保留相对排名)
            cdf_mapped = apply_quantile_mapping(raw_target)
            
            # 【新增】收集非零有效像素用于全局直方图统计
            valid_data = cdf_mapped[cdf_mapped > 0].flatten()
            global_valid_pixels.append(valid_data)

            
            # 保存用于后续神经网络学习的灰度底图
            cdf_save_path = os.path.join(out_dir, "cdf_mapped", proj_name, f"{label}.png")
            save_image(cdf_mapped, cdf_save_path)
            # CDF 已经完美拉平了分布，不需要再被 Gamma 扭曲，原始展示即可 viridis plasma turbo jet
            save_heatmap(cdf_mapped, cdf_save_path.replace("cdf_mapped", "cdf_mapped_vis"), cmap_name="turbo", use_gamma=False)

    print("\n✅ 所有处理完成！这些数据已经完美免疫了极值黑洞与绝对能量抖动，可以直接送入 ConvLSTM 进行训练！")
    

    # 【新增】绘制跨全时间帧的最大误差与平均误差趋势折线图
    if frame_labels:
        import matplotlib.pyplot as plt
        x_ticks = range(len(frame_labels))
        
        # 图1: 全局平均误差趋势对比
        plt.figure(figsize=(14, 7))
        colors = {'tx': 'r', 'ty': 'g', 'tz': 'b'}
        for proj in ['tx', 'ty', 'tz']:
            plt.plot(x_ticks, global_nv_mean_err[proj], color=colors[proj], linestyle=':', alpha=0.6, label=f'Naive Mean ({proj.upper()})')
            plt.plot(x_ticks, global_mg_mean_err[proj], color=colors[proj], marker='.', linewidth=2, label=f'Multi-Gauss Mean ({proj.upper()})')
        plt.title('Global Mean Estimation Error Over Time (Naive vs Multi-Gauss)')
        plt.xlabel('Frame Sequence')
        plt.ylabel('Absolute Error (LogD Domain)')
        plt.xticks(x_ticks[::max(1, len(x_ticks)//10)], [frame_labels[i] for i in x_ticks[::max(1, len(x_ticks)//10)]], rotation=45)
        plt.grid(True, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "global_trend_mean_error.png"), dpi=200)
        plt.close()

        # 图2: 全局最大误差趋势对比
        plt.figure(figsize=(14, 7))
        for proj in ['tx', 'ty', 'tz']:
            plt.plot(x_ticks, global_nv_max_err[proj], color=colors[proj], linestyle=':', alpha=0.6, label=f'Naive Max ({proj.upper()})')
            plt.plot(x_ticks, global_mg_max_err[proj], color=colors[proj], marker='.', linewidth=2, label=f'Multi-Gauss Max ({proj.upper()})')
        plt.title('Global Max Estimation Error Over Time (Naive vs Multi-Gauss)')
        plt.xlabel('Frame Sequence')
        plt.ylabel('Absolute Error (LogD Domain)')
        plt.xticks(x_ticks[::max(1, len(x_ticks)//10)], [frame_labels[i] for i in x_ticks[::max(1, len(x_ticks)//10)]], rotation=45)
        plt.grid(True, alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "global_trend_max_error.png"), dpi=200)
        plt.close()
        
        print("\n📈 全局拟合极值与均值误差折线图已保存。")

    # 【新增】所有循环结束后，绘制全局数据的汇总直方图并计算均匀度指标
    if global_valid_pixels:
        import matplotlib.pyplot as plt
        all_data = np.concatenate(global_valid_pixels)
        
        # 1. 计算具体的数值指标 (基于 50 个区间的概率密度)
        hist_densities, bin_edges = np.histogram(all_data, bins=50, range=(0, 1), density=True)
        # 理论完美均匀分布的密度为 1.0，计算与完美的平均绝对误差 (MAE)
        mae_density = np.mean(np.abs(hist_densities - 1.0))
        # 转换为百分位得分 (100% 代表绝对完美的直线)
        uniformity_score = max(0.0, (1.0 - mae_density) * 100)
        
        print(f"\n📈 全局 CDF 映射质量评估:")
        print(f" - 密度平均绝对误差 (MAE): {mae_density:.4f}")
        print(f" - 数据分布均匀度得分: {uniformity_score:.2f}% (100% 为极端完美状态)")
        
        # 2. 绘制直方图并在图上标注得分
        plt.figure(figsize=(8, 5))
        plt.hist(all_data, bins=50, range=(0,1), density=True, color='teal', alpha=0.7)
        
        # 画一根理论上完美的红线 (y=1.0) 作为辅助对比
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Uniform (y=1.0)')
        
        plt.title(f"Global CDF Uniform Distribution\nUniformity Score: {uniformity_score:.2f}%")
        plt.xlabel("Mapped Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        
        hist_save_path = os.path.join(out_dir, "global_cdf_histogram.png")
        plt.savefig(hist_save_path, dpi=300)
        print(f"📊 全局数据分布直方图已保存至: {hist_save_path}\n")


if __name__ == "__main__":
    if HAS_CUPY:
        print("🚀 已检测到 CuPy，高斯渲染将通过 GPU 加速执行。")
    main()