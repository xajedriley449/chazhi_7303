import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义指数衰减模型 y = A * exp(-\lambda * x) + b
def exp_decay(x, a, lam, b):
    return a * np.exp(-lam * x) + b

def calculate_spatial_ranges(df):
    print("\n--- 空间跨度与倾向性带宽推荐 ---")
    ranges = {}
    recommendations = {}
    for col in ['x', 'y', 'z']:
        if col in df.columns:
            c_min = df[col].min()
            c_max = df[col].max()
            span = c_max - c_min
            ranges[col] = span
            print(f"{col.upper()} 轴跨度: {span:.2f} 米 (Min: {c_min:.2f} -> Max: {c_max:.2f})")
    
    if all(k in ranges for k in ['x', 'y', 'z']):
        # 基于经验：平面扩展占总跨度的 ~5%，垂向衰减更快占总跨度的 ~2.5%
        recommendations['x'] = ranges['x'] * 0.05
        recommendations['y'] = ranges['y'] * 0.05
        recommendations['z'] = ranges['z'] * 0.025
        print("\n💡 根据总跨度比例，推荐的空间高斯带宽 (PHYSICAL_SIGMA_S_METERS) 如下：")
        print(f"X 轴: {recommendations['x']:.2f} 米")
        print(f"Y 轴: {recommendations['y']:.2f} 米")
        print(f"Z 轴: {recommendations['z']:.2f} 米")
        print("注意: 具体值应结合工作面的来压步距与顶底板岩性做最终微调。")
    print("--------------------------------\n")
    return ranges, recommendations

def main():
    csv_path = r"data/7303_xyzet_rotated.csv"
    df = pd.read_csv(csv_path)
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    df['e'] = pd.to_numeric(df['e'], errors='coerce')
    df = df.dropna(subset=['t', 'e']).sort_values('t').reset_index(drop=True)
    
    calculate_spatial_ranges(df)
    
    # 更改 1：使用绝对能量阈值作为主震基准 (例如 10^3 或 10^4 焦耳，请根据实际数据量级调整)
    abs_threshold = 1000.0  
    candidates = df[df['e'] >= abs_threshold].copy()
    
    # 更改 2：主震去重机制 (Declustering)
    # 避免时间上相近的主震发生串扰，若某大事件 12 小时内还有大事件，视为同一破裂群
    main_shocks = []
    last_shock_time = -9999999
    
    for idx, row in candidates.iterrows():
        # 仅当该主震与上一个主震间隔超过 12 小时（或设定的小时数），才作为独立统计事件
        if (row['t'] - last_shock_time) > 12.0 * 3600:
            main_shocks.append(row)
            last_shock_time = row['t']
            
    main_shocks = pd.DataFrame(main_shocks)
    print(f"使用绝对阈值 {abs_threshold} J 且去重后，独立主震组数: {len(main_shocks)}")
    
    observe_hours = 12.0      
    bin_size_hours = 0.5      # 缩小统计粒度到半小时
    bins = np.arange(0, observe_hours + bin_size_hours, bin_size_hours)
    
    all_decay_hist = []
    
    for idx, shock in main_shocks.iterrows():
        t_shock = shock['t']
        aftershocks = df[(df['t'] > t_shock) & (df['t'] <= t_shock + observe_hours * 3600)]
        rel_time_hours = (aftershocks['t'] - t_shock) / 3600.0
        
        # 【修改点】：增加 weights=aftershocks['e']，统计"能量衰减"而不是"频次衰减"
        hist, _ = np.histogram(rel_time_hours, bins=bins, weights=aftershocks['e'])
        all_decay_hist.append(hist)
    
    mean_decay = np.mean(all_decay_hist, axis=0)
    x_data = bins[:-1] + bin_size_hours / 2.0
    y_data = mean_decay
    
    # 指数拟合
    try:
        p0 = [y_data[0], 0.5, np.min(y_data)] # 初始猜测
        popt, _ = curve_fit(exp_decay, x_data, y_data, p0=p0, maxfev=10000)
        a, lam, b = popt
        
        # 计算降至 1/e 处的时间（相较于峰值幅值）
        # a * exp(-lam * t) = a * exp(-1) => t = 1 / lam
        estimated_sigma = 1.0 / lam
        print(f"数据拟合指数衰减率 λ = {lam:.4f}")
        print(f"均值频次衰减至峰值的 1/e 约需: {estimated_sigma:.2f} 小时。")
    except Exception as e:
        print("曲线拟合失败，采用最值查找法:", e)
        target_val = y_data.max() * np.exp(-1)
        decay_idx = np.where(y_data < target_val)[0]
        estimated_sigma = x_data[decay_idx[0]] if len(decay_idx) > 0 else 4.0
        print(f"退化计算得出衰减至 1/e 需: {estimated_sigma:.2f} 小时。")

    # 绘图
    plt.figure(figsize=(10, 6))
    # 【修改点】：将 label='实测频次均值' 改为 label='实测能量均值'
    plt.bar(bins[:-1], mean_decay, width=bin_size_hours*0.8, align='edge', alpha=0.5, color='steelblue', label='实测能量均值')
    
    # 绘制拟合曲线
    if 'popt' in locals():
        x_fit = np.linspace(0, observe_hours, 100)
        y_fit = exp_decay(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'指数拟合 ($y={a:.2f}e^{{-{lam:.2f}t}}+{b:.2f}$)')
        plt.axvline(estimated_sigma, color='g', linestyle='-.', label=f'时间带宽 {estimated_sigma:.2f} h')

    plt.xlabel('主震后时间 (h)')
    # 【修改点】：Y轴标签更新
    plt.ylabel('平均释放能量 (J)')
    #plt.title('大能量微震事件后能量释放衰减统计')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 【修改点】：添加保存图片的逻辑
    save_dir = os.path.join('another', 'calculate_sigma')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'energy_decay_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"统计图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()