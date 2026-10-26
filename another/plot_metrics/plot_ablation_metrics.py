import os
import matplotlib.pyplot as plt
import numpy as np

# 1. 设置学术论文标准字体：英文 Times New Roman，中文 宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文宋体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# 强制英文和数字使用 Times New Roman
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix' # 接近 Times New Roman 的数学字体

def parse_results(file_path):
    """解析结果文件，提取各视角的指标"""
    results = {}
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return results

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    views = ['tx', 'ty', 'tz', 'xy', 'xz', 'yz']
    current_view = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        for view in views:
            if line.endswith(f" {view}"):
                if len(line.split()) == 2:
                    current_view = view.upper()
                break
                
        if current_view:
            if current_view not in results:
                results[current_view] = {}
            
            if line.startswith('平均 SSIM:'):
                results[current_view]['SSIM'] = float(line.split(':')[1].strip())
            elif line.startswith('平均 PSNR:'):
                results[current_view]['PSNR'] = float(line.split(':')[1].strip())
            elif line.startswith('平均 MSE :'):
                results[current_view]['MSE'] = float(line.split(':')[1].strip())
            elif line.startswith('平均 MAE :'):
                results[current_view]['MAE'] = float(line.split(':')[1].strip())
                current_view = None 

    return results

def main():
    files = {
        'E_data': 'output_txyz/e_data_result.txt',
        'Cov1': 'output_txyz/t_cov1_result.txt',
        'Cov2': 'output_txyz/xyz_cov2_result.txt',
        'Cov3': 'output_txyz/cov3_result.txt'
    }

    all_data = {}
    for label, path in files.items():
        all_data[label] = parse_results(path)

    views = ['TX', 'TY', 'TZ', 'XY', 'XZ', 'YZ']
    metrics = ['SSIM', 'PSNR', 'MSE', 'MAE']
    
    plot_data = {metric: {label: [] for label in files.keys()} for metric in metrics}
    
    for metric in metrics:
        for label in files.keys():
            for view in views:
                val = all_data[label].get(view, {}).get(metric, 0)
                plot_data[metric][label].append(val)

    x = np.arange(len(views))
    width = 0.2
    
    # 论文级配色 (去掉纹理，保持简洁，颜色对比度高)
    colors = ['#E0E0E0', '#A9CCE3', '#4682B4', '#E74C3C']

    # 分别为 4 个指标生成独立的图表
    for metric in metrics:
        # 【修改点1】调整比例为 6:5，更适合并排
        fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
        
        for j, (label, color) in enumerate(zip(files.keys(), colors)):
            offset = (j - 1.5) * width
            # 【修改点2】稍微加粗柱子边框
            ax.bar(x + offset, plot_data[metric][label], width, 
                   label=label, color=color, edgecolor='black', linewidth=1.0, zorder=3)
            
        # 【修改点3】大幅放大坐标轴刻度字体 (12 -> 16)
        ax.set_xticks(x)
        ax.set_xticklabels(views, fontsize=16, fontname='Times New Roman')
        ax.tick_params(axis='y', labelsize=16)
        
        # 【修改点4】大幅放大 Y 轴标题字体 (12 -> 18)
        ax.set_ylabel(metric, fontsize=18, fontname='Times New Roman', fontweight='bold')
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        
        # 针对不同指标微调 Y 轴范围
        if metric == 'SSIM':
            ax.set_ylim(0.75, 1.0)
        elif metric == 'PSNR':
            ax.set_ylim(20, 35)
            
        # 【修改点5】放大图例字体 (10.5 -> 14)
        ax.legend(loc='best', prop={'family': 'SimSun', 'size': 14}, framealpha=0.9)

        # 【修改点6】加粗坐标轴边框，防止缩小后看不清
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        plt.tight_layout()
        
        # 保存为独立的图片
        save_path = rf'another\plot_metrics\ablation_metric_{metric}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"学术标准图表(并排优化版)已成功生成并保存至: {save_path}")
        
        # 关闭当前图表，释放内存
        plt.close(fig)

if __name__ == '__main__':
    main()
