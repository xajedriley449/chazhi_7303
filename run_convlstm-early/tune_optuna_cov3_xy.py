import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 【强制修复】：禁止底层 Fortran MKL 库截获 Ctrl+C 信号，把控制权还给 Python
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import glob
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna

from loss import SharpCompositeLoss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 极速版三通道显存数据集 ====================
class MemTripletSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, main_paths: list[str], cov1_map: dict[str, str], cov2_map: dict[str, str], seq_len: int, device='cpu'):
        self.seq_len = seq_len
        
        kept = []
        for p in main_paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if name in cov1_map and name in cov2_map:
                kept.append(p)
                
        print("正在将 TX 和 且包含 COV1、COV2 的图像(NPY)同时直接加载到【GPU显存】中...")
        m_list, c1_list, c2_list = [], [], []
        
        for p in kept:
            name = os.path.splitext(os.path.basename(p))[0]
            
            # tx
            arr_m = np.load(p).astype(np.float32)
            if arr_m.max() > arr_m.min():
                arr_m = (arr_m - arr_m.min()) / (arr_m.max() - arr_m.min())
            else: arr_m = np.zeros_like(arr_m)
            m_list.append(torch.tensor(arr_m).unsqueeze(0))
            
            # cov1
            arr_c1 = np.load(cov1_map[name]).astype(np.float32)
            if arr_c1.max() > arr_c1.min():
                arr_c1 = (arr_c1 - arr_c1.min()) / (arr_c1.max() - arr_c1.min())
            else: arr_c1 = np.zeros_like(arr_c1)
            c1_list.append(torch.tensor(arr_c1).unsqueeze(0))
            
            # cov2
            arr_c2 = np.load(cov2_map[name]).astype(np.float32)
            if arr_c2.max() > arr_c2.min():
                arr_c2 = (arr_c2 - arr_c2.min()) / (arr_c2.max() - arr_c2.min())
            else: arr_c2 = np.zeros_like(arr_c2)
            c2_list.append(torch.tensor(arr_c2).unsqueeze(0))
            
        self.all_main = torch.stack(m_list, dim=0).to(device)
        self.all_cov1 = torch.stack(c1_list, dim=0).to(device)
        self.all_cov2 = torch.stack(c2_list, dim=0).to(device)
        self.num_frames = len(self.all_main)
        print(f"显存预加载完成！已在GPU驻留 {self.num_frames} 帧三通道对。")

    def __len__(self):
        return max(0, self.num_frames - self.seq_len)

    def __getitem__(self, idx: int):
        seq_tx = self.all_main[idx: idx + self.seq_len]    
        seq_cov1 = self.all_cov1[idx: idx + self.seq_len]  
        seq_cov2 = self.all_cov2[idx: idx + self.seq_len]  
        
        x_seq = torch.cat([seq_tx, seq_cov1, seq_cov2], dim=1)
        y = self.all_main[idx + self.seq_len]                  
        return x_seq, y

# ==================== ConvLSTM 核心 ====================
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
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_state, c_state = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h_state, c_state]
                )
                output_inner.append(h_state)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])
        
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

# ==================== 早期融合预测模型 ====================
class EarlyFusionConvLSTMPredictor(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=32, num_layers=2):
        super(EarlyFusionConvLSTMPredictor, self).__init__()
        
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

# ==================== 全局数据加载器 (避免重复加载) ====================
GLOBAL_TRAIN_LOADER = None
GLOBAL_VAL_LOADER = None

# ==================== Optuna 目标函数 ====================
def objective(trial):
    # 1. 定义搜索空间
    hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 构建模型
    model = EarlyFusionConvLSTMPredictor(
        input_channels=3, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = SharpCompositeLoss(l1_weight=1.0, gdl_weight=5.0, energy_factor=15.0)
    
    # 调参时可以适当减少 epochs，或者依赖 Optuna 的 Pruner 提前终止
    epochs = 30 
    
    for epoch in range(epochs):
        model.train()
        for seq, target in GLOBAL_TRAIN_LOADER:
            optimizer.zero_grad(set_to_none=True)
            output = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in GLOBAL_VAL_LOADER:
                output = model(seq)
                val_loss += criterion(output, target).item()
        val_loss /= len(GLOBAL_VAL_LOADER)
        
        # 向 Optuna 报告当前 epoch 的验证集 Loss
        trial.report(val_loss, epoch)
        
        # 如果 Optuna 认为这组参数没希望了，就提前剪枝（终止）
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return val_loss

# ==================== 主函数 ====================
def main():
    set_seed(42)
    global GLOBAL_TRAIN_LOADER, GLOBAL_VAL_LOADER
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    wp = "xy"
    seq_len = 10
    batch_size = 4
    
    data_dir = rf"input_txyz\e_data\matrix_npy\{wp}"
    cov1_dir = rf"input_txyz\t_cov1\matrix_npy\{wp}"
    cov2_dir = rf"input_txyz\xyz_cov2\matrix_npy\{wp}"
    
    main_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    cov1_paths = sorted(glob.glob(os.path.join(cov1_dir, "*.npy")))
    cov2_paths = sorted(glob.glob(os.path.join(cov2_dir, "*.npy")))
    
    if not main_paths or not cov1_paths or not cov2_paths:
        print("缺少图片，请检查路径！")
        return
        
    cov1_map = {os.path.splitext(os.path.basename(p))[0]: p for p in cov1_paths}
    cov2_map = {os.path.splitext(os.path.basename(p))[0]: p for p in cov2_paths}
    
    # 划分训练集和测试集 (调参时我们只用训练集划分出的 train/val)
    split_idx = int(0.8 * len(main_paths))
    train_main = main_paths[:split_idx]
    
    full_train_dataset = MemTripletSequenceDataset(train_main, cov1_map, cov2_map, seq_len, device=device)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    GLOBAL_TRAIN_LOADER = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    GLOBAL_VAL_LOADER = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # 创建 Optuna Study
    # 使用 MedianPruner：如果当前 trial 的中间结果比之前所有 trial 的中位数还差，就剪枝
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    
    print(f"\n🚀 开始在 {wp} 视角上进行 Optuna 超参数搜索...")
    # 运行 30 次尝试 (您可以根据时间调整 n_trials)
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*50)
    print("🏆 搜索完成！最佳超参数组合如下：")
    print("="*50)
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print(f"最佳验证集 Loss: {study.best_value:.6f}")
    print("="*50)
    
    # 保存结果
    os.makedirs("output_txyz/optuna_results", exist_ok=True)
    with open("output_txyz/optuna_results/best_params_xy.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("✅ 最佳参数已保存至 output_txyz/optuna_results/best_params_xy.json")

if __name__ == "__main__":
    main()
