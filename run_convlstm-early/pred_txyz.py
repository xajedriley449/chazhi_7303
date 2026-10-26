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

# ==================== 特效：能量加权损失函数 ====================
class EnergyWeightedLoss(nn.Module):
    def __init__(self, weight_factor=10.0):
        super().__init__()
        self.weight_factor = weight_factor

    def forward(self, pred, target):
        base_loss = torch.nn.functional.l1_loss(pred, target, reduction='none')
        weight_mask = 1.0 + target * self.weight_factor
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

# ==================== ConvLSTM Cell ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, _, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            hx, cx = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                hx, cx = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[hx, cx]
                )
                output_inner.append(hx)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([hx, cx])
        
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# ==================== 简单的预测模型 ====================
class SimpleConvLSTMPredictor(nn.Module):
    """最简单的ConvLSTM预测模型"""
    def __init__(self, input_channels=1, hidden_dim=64, num_layers=1):
        super(SimpleConvLSTMPredictor, self).__init__()
        
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, x):
        layer_output, last_state = self.convlstm(x)
        h = last_state[0][0]
        output = self.output_conv(h)
        return torch.sigmoid(output)

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
        # self.all_x 切片出来的维度本身已经是 [seq_len, 1, H, W]
        x_seq = self.all_x[idx : idx + self.seq_len] 
        # y 切片出来的维度本身已经是 [1, H, W]
        y = self.all_x[idx + self.seq_len]           
        return x_seq, y

# ==================== 训练函数 ====================
def train(model, train_loader, val_loader, epochs, device, output_dir, learning_rate=1e-3):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    #criterion = EnergyWeightedLoss(weight_factor=15.0)
        # 【修改】：使用新的复合锐化损失函数
    criterion = SharpCompositeLoss(l1_weight=1.0, gdl_weight=5.0, energy_factor=15.0)  
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
    )
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    # 【新增】：早停法参数
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
            
    # 修改返回值，增加 stopped_epoch 和 best_epoch (即 stopped_epoch - patience)
    best_epoch = stopped_epoch - epochs_no_improve if 'stopped_epoch' in locals() else epochs
    return train_losses, val_losses, best_epoch

# ==================== 评估函数 ====================
def evaluate(model, test_dataset, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建多格式数据存放目录
    d_npy = os.path.join(output_dir, "matrix_npy")
    d_raw = os.path.join(output_dir, "raw_interp")
    d_raw_vis = os.path.join(output_dir, "raw_interp_vis")
    d_cdf = os.path.join(output_dir, "cdf_mapped")
    d_cdf_vis = os.path.join(output_dir, "cdf_mapped_vis")
    d_pred = os.path.join(output_dir, "pred_images")
    
    for d in [d_npy, d_raw, d_raw_vis, d_cdf, d_cdf_vis, d_pred]:
        os.makedirs(d, exist_ok=True)
    
    # 新增指标列表
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
            
            # 计算 SSIM
            score_ssim = ssim(target_np, pred, data_range=1.0)
            ssim_scores.append(score_ssim)
            
            # 计算额外的指标 (MSE, MAE, PSNR)
            mse_val = np.mean((target_np - pred) ** 2)
            mae_val = np.mean(np.abs(target_np - pred))
            # 限制极小值防止除以 0
            psnr_val = 10 * np.log10(1.0 / mse_val) if mse_val > 1e-10 else 100.0
            
            mse_scores.append(mse_val)
            mae_scores.append(mae_val)
            psnr_scores.append(psnr_val)
            
            # =============== 保存多类型格式数据 ===============
            label = f"pred_frame_{idx:03d}"
            
            # 1. 保存 matrix_npy
            np.save(os.path.join(d_npy, f"{label}.npy"), pred)
            
            # 2. 保存 raw_interp / raw_interp_vis
            save_image(pred, os.path.join(d_raw, f"{label}.png"))
            save_heatmap(pred, os.path.join(d_raw_vis, f"{label}.png"), use_gamma=True)
            
            # 3. 计算并保存 CDF 映射图
            cdf_mapped = apply_quantile_mapping(pred)
            save_image(cdf_mapped, os.path.join(d_cdf, f"{label}.png"))
            save_heatmap(cdf_mapped, os.path.join(d_cdf_vis, f"{label}.png"), cmap_name="turbo", use_gamma=False)
            
            # =============== 原对比绘图逻辑 (修改为 Magma 色带) =================
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 使用 magma 色带展示能量场，使用 gamma 校正让低能量区域更清晰
            axes[0].imshow(target_np ** 0.3, cmap='magma', vmin=0, vmax=1)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            axes[1].imshow(pred ** 0.3, cmap='magma', vmin=0, vmax=1)
            axes[1].set_title(f'Prediction (SSIM: {score_ssim:.4f})')
            axes[1].axis('off')
            
            # 误差图使用 hot
            im3 = axes[2].imshow(np.abs(pred - target_np), cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'Abs Diff (MSE: {mse_val:.4f})')
            axes[2].axis('off')
            
            # 自动调整子图参数，增加边缘留白，防止标题被切掉
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
    start_time = time.time() # 记录开始时间
    
    print(f"\n{'='*50}")
    print(f"🚀 开始执行视角: {wp} 的预测任务")
    print(f"{'='*50}")
    
    config = {
        "seq_len": 10,
        "batch_size": 4,      # 所有人必须一致！向显存占用最大的 cov3 妥协
        "epochs": 60,
        "hidden_dim": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
        "data_dir": rf"input_txyz\e_data\matrix_npy\{wp}"
    }
    
    output_dir = rf"output_txyz/e_data/{wp}_pred_npy_convlstm"
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
    
    model = SimpleConvLSTMPredictor(
        input_channels=1, 
        hidden_dim=config["hidden_dim"], 
        num_layers=config["num_layers"]
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    end_time = time.time() # 记录结束时间
    total_time_seconds = end_time - start_time
    hours, rem = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"

    with open(os.path.join(output_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=============== 实验结果 ===============\n")
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
    
    # 定义需要批量运行的视角列表
    wp_list = ["xy"]#, "xz", "yz", "tx", "ty", "tz"] 
    
    for wp in wp_list:
        run_experiment(wp, device)
        
        # 每次实验结束后，清理显存，防止 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 已清理 GPU 显存。")

if __name__ == "__main__":
    main()
