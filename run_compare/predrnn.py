import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 【强制修复】：禁止底层 Fortran MKL 库截获 Ctrl+C 信号，把控制权还给 Python
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import time
import glob
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from img_tools import save_image, save_heatmap, apply_quantile_mapping
from loss import SharpCompositeLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 极速版显存数据集 ====================
class SequenceDataset(Dataset):
    def __init__(self, image_paths, seq_len=5, device='cpu'):
        self.seq_len = seq_len
        
        print("正在将所有图像(NPY)直接加载到【GPU显存】中... 这将根除所有传输延迟！")
        m_list = []
        for p in image_paths:
            arr = np.load(p).astype(np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr)
            tensor_arr = torch.tensor(arr).unsqueeze(0)
            m_list.append(tensor_arr)
            
        self.all_x = torch.stack(m_list, dim=0).to(device)
        self.num_frames = len(self.all_x)
        print(f"显存预加载完成！已在GPU驻留 {self.num_frames} 帧。")

    def __len__(self):
        return max(0, self.num_frames - self.seq_len)
    
    def __getitem__(self, idx):
        x_seq = self.all_x[idx : idx + self.seq_len] 
        y = self.all_x[idx + self.seq_len]           
        return x_seq, y

# ==================== PredRNN 核心组件 ====================
class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 7, width, width]) if layer_norm else nn.Identity()
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, width, width]) if layer_norm else nn.Identity()
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 3, width, width]) if layer_norm else nn.Identity()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, width, width]) if layer_norm else nn.Identity()
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new

class PredRNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(PredRNN, self).__init__()
        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        cell_list = []
        width = configs['img_width']
        
        for i in range(num_layers):
            in_channel = configs['in_channel'] if i == 0 else num_hidden[i-1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs['filter_size'],
                                       configs['stride'], configs['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true=None):
        batch = frames_tensor.shape[0]
        seq_len = frames_tensor.shape[1]
        height = frames_tensor.shape[3]
        width = frames_tensor.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(frames_tensor.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(frames_tensor.device)

        for t in range(seq_len):
            net = frames_tensor[:, t]
            
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        return torch.sigmoid(next_frames[-1])

# ==================== 训练函数 ====================
def train(model, train_loader, val_loader, epochs, device, output_dir, learning_rate=1e-3):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = SharpCompositeLoss(l1_weight=1.0, gdl_weight=5.0, energy_factor=15.0)  
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    early_stop_patience = 15  
    epochs_no_improve = 0   
    
    print("\n[提示] 训练途中随时可按 Ctrl+C 中止训练，系统将自动加载最优模型并进入扫尾阶段...")
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for seq, target in train_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seq, target in val_loader:
                    output = model(seq)
                    val_loss += criterion(output, target).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f} | Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                epochs_no_improve = 0  
            else:
                epochs_no_improve += 1 
                
            if epochs_no_improve >= early_stop_patience:
                print(f"\n🛑 触发早停机制！验证集 Loss 已经连续 {early_stop_patience} 个 Epoch 没有改善。")
                stopped_epoch = epoch + 1
                break 
                
    except KeyboardInterrupt:
        print("\n\n[指令] 收到了手动中断信号 (Ctrl+C)！")
        print(f"训练在 Epoch {len(train_losses)} 提前终止。正在进入扫尾工作...")
        stopped_epoch = len(train_losses)
        
    finally:
        best_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path))
            print("✅ 成功加载了迄今为止最好的模型权重 (best_model.pth)。")
        else:
            print("⚠️ 警告：还未完成1个Epoch就被中断，没有任何模型被保存。")
            
    best_epoch = stopped_epoch - epochs_no_improve if 'stopped_epoch' in locals() else epochs
    return train_losses, val_losses, best_epoch

# ==================== 评估函数 ====================
def evaluate(model, test_dataset, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    d_npy = os.path.join(output_dir, "matrix_npy")
    d_raw = os.path.join(output_dir, "raw_interp")
    d_raw_vis = os.path.join(output_dir, "raw_interp_vis")
    d_cdf = os.path.join(output_dir, "cdf_mapped")
    d_cdf_vis = os.path.join(output_dir, "cdf_mapped_vis")
    d_pred = os.path.join(output_dir, "pred_images")
    
    for d in [d_npy, d_raw, d_raw_vis, d_cdf, d_cdf_vis, d_pred]:
        os.makedirs(d, exist_ok=True)
    
    ssim_scores = []
    mse_scores = []
    mae_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            seq, target = test_dataset[idx]
            seq = seq.unsqueeze(0).to(device)
            
            pred = model(seq).cpu().squeeze().numpy()
            target_np = target.cpu().squeeze().numpy()
            
            score_ssim = ssim(target_np, pred, data_range=1.0)
            ssim_scores.append(score_ssim)
            
            mse_val = np.mean((target_np - pred) ** 2)
            mae_val = np.mean(np.abs(target_np - pred))
            psnr_val = 10 * np.log10(1.0 / mse_val) if mse_val > 1e-10 else 100.0
            
            mse_scores.append(mse_val)
            mae_scores.append(mae_val)
            psnr_scores.append(psnr_val)
            
            label = f"pred_frame_{idx:03d}"
            
            np.save(os.path.join(d_npy, f"{label}.npy"), pred)
            save_image(pred, os.path.join(d_raw, f"{label}.png"))
            save_heatmap(pred, os.path.join(d_raw_vis, f"{label}.png"), use_gamma=True)
            
            cdf_mapped = apply_quantile_mapping(pred)
            save_image(cdf_mapped, os.path.join(d_cdf, f"{label}.png"))
            save_heatmap(cdf_mapped, os.path.join(d_cdf_vis, f"{label}.png"), cmap_name="turbo", use_gamma=False)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(target_np ** 0.3, cmap='magma', vmin=0, vmax=1)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            axes[1].imshow(pred ** 0.3, cmap='magma', vmin=0, vmax=1)
            axes[1].set_title(f'Prediction (SSIM: {score_ssim:.4f})')
            axes[1].axis('off')
            
            im3 = axes[2].imshow(np.abs(pred - target_np), cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'Abs Diff (MSE: {mse_val:.4f})')
            axes[2].axis('off')
            
            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(d_pred, f"compare_{idx:03d}.png"))
            plt.close()
            
    print(f"平均 SSIM: {np.mean(ssim_scores):.4f} | 平均 PSNR: {np.mean(psnr_scores):.4f}")
    print(f"平均 MSE:  {np.mean(mse_scores):.6f} | 平均 MAE:  {np.mean(mae_scores):.6f}")
    
    base_dir = os.path.dirname(output_dir)
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(ssim_scores)), ssim_scores, marker='o', linewidth=2, markersize=5)
    plt.axhline(y=np.mean(ssim_scores), color='red', linestyle='--', label=f'Mean SSIM: {np.mean(ssim_scores):.4f}')
    plt.xlabel('Frame Index', fontsize=12); plt.ylabel('SSIM Score', fontsize=12)
    plt.title('SSIM Score per Frame', fontsize=14, fontweight='bold')
    plt.ylim([0, 1]); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'ssim_curve.png'), dpi=150)
    plt.close()
    
    return {
        'ssim': ssim_scores,
        'mse': mse_scores,
        'mae': mae_scores,
        'psnr': psnr_scores
    }

# ==================== 单次实验运行函数 ====================
def run_experiment(wp, device):
    start_time = time.time() 
    
    print(f"\n{'='*50}")
    print(f"🚀 开始执行视角: {wp} 的预测任务 (PredRNN)")
    print(f"{'='*50}")
    
    config = {
        "seq_len": 10,
        "batch_size": 4,      
        "epochs": 60,
        "hidden_dim": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
        "data_dir": rf"input_txyz\e_data\matrix_npy\{wp}"
    }
    
    output_dir = rf"output_txyz_cp/e_data/{wp}_pred_npy_predrnn"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    image_paths = sorted(glob.glob(os.path.join(config["data_dir"], "*.npy")))
    print(f"Found {len(image_paths)} images (npy format) for {wp}")
    
    if len(image_paths) == 0:
        print(f"未找到图片，请检查 data_dir: {config['data_dir']}！跳过当前视角。")
        return

    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    test_paths = image_paths[split_idx - config["seq_len"]:]
    
    full_train_dataset = SequenceDataset(train_paths, config["seq_len"], device=device)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    test_dataset = SequenceDataset(test_paths, config["seq_len"], device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    
    # 【修改】：使用 PredRNN 模型，输入通道为 1
    predrnn_configs = {
        'in_channel': 1,
        'img_width': 128, # 假设图像大小为 64x64，如果不是请修改
        'filter_size': 3,
        'stride': 1,
        'layer_norm': False
    }
    model = PredRNN(num_layers=config["num_layers"], num_hidden=[config["hidden_dim"]]*config["num_layers"], configs=predrnn_configs)
    print(f"PredRNN（单输入）模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n开始训练 {wp}...")
    train_losses, val_losses, best_epoch = train(
        model, train_loader, val_loader, 
        config["epochs"], device, output_dir, config["learning_rate"]
    )
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'last_epoch_model.pth'))
    
    with open(os.path.join(output_dir, 'loss_log.csv'), 'w', encoding='utf-8') as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (t_l, v_l) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{t_l:.6f},{v_l:.6f}\n")
    
    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss.png')); plt.close()
    
    print(f"\n评估测试集 {wp}...")
    metrics = evaluate(model, test_dataset, device, os.path.join(output_dir, 'predictions'))
    
    end_time = time.time() 
    total_time_seconds = end_time - start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"

    with open(os.path.join(output_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=============== PredRNN 实验结果 ===============\n")
        f.write(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"测试集帧数: {len(metrics['ssim'])}\n")
        f.write(f"总运行时长: {time_str}\n")
        f.write(f"最佳模型 Epoch: {best_epoch} (总设定 Epochs: {config['epochs']})\n") 
        if metrics['ssim']:
            f.write(f"平均 SSIM:  {np.mean(metrics['ssim']):.4f}\n")
            f.write(f"平均 PSNR:  {np.mean(metrics['psnr']):.4f}\n")
            f.write(f"平均 MSE :  {np.mean(metrics['mse']):.6f}\n")
            f.write(f"平均 MAE :  {np.mean(metrics['mae']):.6f}\n")
        f.write("======================================\n")

    print(f"\n✅ {wp} 视角处理完成！耗时: {time_str}。结果保存在: {output_dir}")

# ==================== 主函数 ====================
def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    wp_list = ["xy"]#, "xz", "yz", "tx", "ty", "tz"] 
    
    for wp in wp_list:
        run_experiment(wp, device)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 已清理 GPU 显存。")

if __name__ == "__main__":
    main()
