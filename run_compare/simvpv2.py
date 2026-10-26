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

# ==================== SimVPv2 核心组件 ====================
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride // 2)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gSTA(nn.Module):
    def __init__(self, dim, kernel_size=11):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        # 大核深度可分离卷积 (Large-kernel Depthwise Conv)
        self.spatial_gating_unit = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = x * shortcut  # 门控机制 (Gating)
        x = self.proj_2(x)
        return x

class GASubBlock(nn.Module):
    def __init__(self, dim, kernel_size=11, mlp_ratio=4., drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = gSTA(dim, kernel_size=kernel_size)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimVPv2(nn.Module):
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=4, kernel_size=11):
        super(SimVPv2, self).__init__()
        T, C, H, W = shape_in
        self.enc = nn.Sequential(
            ConvSC(C, hid_S, stride=1, act_norm=True),
            *[ConvSC(hid_S, hid_S, stride=1, act_norm=True) for _ in range(N_S - 1)]
        )
        
        # Translator 升级为 gSTA (GASubBlock)
        self.hid = nn.Sequential(
            ConvSC(T * hid_S, hid_T, stride=1, act_norm=True),
            *[GASubBlock(hid_T, kernel_size=kernel_size) for _ in range(N_T)],
            ConvSC(hid_T, T * hid_S, stride=1, act_norm=False)
        )
        
        self.dec = nn.Sequential(
            *[ConvSC(hid_S, hid_S, stride=1, act_norm=True) for _ in range(N_S - 1)],
            ConvSC(hid_S, hid_S, stride=1, act_norm=True)
        )
        
        self.readout = nn.Conv2d(hid_S, 1, 1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x = x_seq.view(B * T, C, H, W)

        embed = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z.reshape(B, T * C_, H_, W_))
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid)
        Y = self.readout(Y)
        Y = Y.view(B, T, 1, H, W)
        
        return torch.sigmoid(Y[:, -1, :, :, :])

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
    print(f"🚀 开始执行视角: {wp} 的预测任务 (SimVPv2)")
    print(f"{'='*50}")
    
    config = {
        "seq_len": 10,
        "batch_size": 4,      
        "epochs": 60,
        "learning_rate": 1e-3,
        "data_dir": rf"input_txyz\e_data\matrix_npy\{wp}"
    }
    
    output_dir = rf"output_txyz_cp/e_data/{wp}_pred_npy_simvpv2"
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
    
    # 【修改】：使用 SimVPv2 模型，输入通道为 1
    shape_in = (config["seq_len"], 1, 128, 128) # 假设图像大小为 128x128
    model = SimVPv2(shape_in=shape_in, hid_S=16, hid_T=128, N_S=2, N_T=2, kernel_size=11)
    print(f"SimVPv2（单输入）模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
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
        f.write("=============== SimVPv2 实验结果 ===============\n")
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
