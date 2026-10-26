import os
import sys
import time
import glob
import random
import json
from pathlib import Path

# Resolve project paths relative to this script, not the shell's current directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Let Python handle Ctrl+C instead of the underlying Fortran library.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from img_tools import save_image, save_heatmap, apply_quantile_mapping
from loss import SharpCompositeLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_frame(path):
    """Load one 2D matrix and independently min-max normalize it to [0, 1]."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"预期二维矩阵，实际形状为 {arr.shape}: {path}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


# ==================== 双通道序列数据集 ====================
class MemPairedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, paired_paths, seq_len, device="cpu"):
        self.seq_len = seq_len
        main_frames = []
        cov_frames = []
        expected_shape = None

        print(f"正在将能量场与红线协变量场加载到 {device}...")
        for main_path, cov_path in paired_paths:
            main_arr = normalize_frame(main_path)
            cov_arr = normalize_frame(cov_path)

            if main_arr.shape != cov_arr.shape:
                raise ValueError(
                    "能量场与红线协变量场矩阵尺寸不一致：\n"
                    f"  energy: {main_path} {main_arr.shape}\n"
                    f"  red_cov: {cov_path} {cov_arr.shape}"
                )
            if expected_shape is None:
                expected_shape = main_arr.shape
            elif main_arr.shape != expected_shape:
                raise ValueError(
                    f"同一视角下矩阵尺寸不统一：{main_path} {main_arr.shape}，"
                    f"预期 {expected_shape}"
                )

            main_frames.append(torch.from_numpy(main_arr).unsqueeze(0))
            cov_frames.append(torch.from_numpy(cov_arr).unsqueeze(0))

        if not main_frames:
            raise ValueError("没有可加载的能量场/红线协变量配对数据。")

        self.all_main = torch.stack(main_frames, dim=0).to(device)
        self.all_cov = torch.stack(cov_frames, dim=0).to(device)
        self.num_frames = len(self.all_main)
        print(f"已加载 {self.num_frames} 帧双通道配对数据。")

    def __len__(self):
        # Each sample uses seq_len input frames and the following frame as target.
        return max(0, self.num_frames - self.seq_len)

    def __getitem__(self, idx):
        seq_energy = self.all_main[idx : idx + self.seq_len]
        seq_red_cov = self.all_cov[idx : idx + self.seq_len]

        # Early fusion: concatenate energy and red_cov along the channel axis.
        x_seq = torch.cat([seq_energy, seq_red_cov], dim=1)
        target = self.all_main[idx + self.seq_len]
        return x_seq, target


# ==================== ConvLSTM ====================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        weight = self.conv.weight
        return (
            weight.new_zeros(batch_size, self.hidden_dim, height, width),
            weight.new_zeros(batch_size, self.hidden_dim, height, width),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=True,
        bias=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim] * num_layers
        else:
            hidden_dims = hidden_dim
        if isinstance(kernel_size, tuple):
            kernel_sizes = [kernel_size] * num_layers
        else:
            kernel_sizes = kernel_size

        if len(hidden_dims) != num_layers or len(kernel_sizes) != num_layers:
            raise ValueError("hidden_dim/kernel_size 的层数必须与 num_layers 一致。")

        cells = []
        for layer_idx in range(num_layers):
            layer_input_dim = input_dim if layer_idx == 0 else hidden_dims[layer_idx - 1]
            cells.append(
                ConvLSTMCell(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dims[layer_idx],
                    kernel_size=kernel_sizes[layer_idx],
                    bias=bias,
                )
            )
        self.cell_list = nn.ModuleList(cells)

    def forward(self, input_tensor):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.shape
        layer_input = input_tensor
        last_states = []

        for layer_idx, cell in enumerate(self.cell_list):
            h_state, c_state = cell.init_hidden(
                batch_size, (height, width)
            )
            outputs = []
            for t in range(seq_len):
                h_state, c_state = cell(
                    layer_input[:, t], (h_state, c_state)
                )
                outputs.append(h_state)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append((h_state, c_state))

        return layer_input, last_states


class EarlyFusionConvLSTMPredictor(nn.Module):
    """Predict the next energy field from energy + red_cov input channels."""
    def __init__(self, input_channels=2, hidden_dim=32, num_layers=2):
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            bias=True,
        )
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        _, last_states = self.convlstm(x)
        final_hidden = last_states[-1][0]
        return torch.sigmoid(self.output_conv(final_hidden))


# ==================== Training ====================
def train(model, train_loader, val_loader, epochs, device, output_dir, learning_rate):
    if len(train_loader) == 0:
        raise ValueError("训练 DataLoader 为空，请检查数据量和 batch_size。")
    if len(val_loader) == 0:
        raise ValueError("验证 DataLoader 为空，请检查训练/验证划分。")

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    criterion = SharpCompositeLoss(
        l1_weight=1.0,
        gdl_weight=5.0,
        energy_factor=15.0,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_patience = 15
    epochs_no_improve = 0

    print("\n训练过程中可按 Ctrl+C 中断；中断后会尝试加载最佳验证模型。")
    try:
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0

            for seq, target in train_loader:
                seq = seq.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(seq)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

            train_loss = train_loss_sum / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for seq, target in val_loader:
                    seq = seq.to(device)
                    target = target.to(device)
                    val_loss_sum += criterion(model(seq), target).item()

            val_loss = val_loss_sum / len(val_loader)
            val_losses.append(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{epochs}] LR: {current_lr:.6g} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, "best_model.pth"),
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(
                    f"触发早停：验证损失连续 {early_stop_patience} 个 epoch 未改善。"
                )
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，结束训练并进行收尾。")
    finally:
        best_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"已加载最佳模型，最佳 epoch: {best_epoch}。")
        else:
            print("没有保存到最佳模型；训练可能在第一个 epoch 完成前中断。")

    return train_losses, val_losses, best_epoch


# ==================== Evaluation ====================
def evaluate(model, test_dataset, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    dirs = {
        "npy": os.path.join(output_dir, "matrix_npy"),
        "raw": os.path.join(output_dir, "raw_interp"),
        "raw_vis": os.path.join(output_dir, "raw_interp_vis"),
        "cdf": os.path.join(output_dir, "cdf_mapped"),
        "cdf_vis": os.path.join(output_dir, "cdf_mapped_vis"),
        "compare": os.path.join(output_dir, "pred_images"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    scores = {"ssim": [], "mse": [], "mae": [], "psnr": []}

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            seq, target = test_dataset[idx]
            prediction = model(seq.unsqueeze(0).to(device))
            pred = prediction[0, 0].cpu().numpy()
            target_np = target[0].cpu().numpy()

            score_ssim = ssim(target_np, pred, data_range=1.0)
            mse_val = float(np.mean((target_np - pred) ** 2))
            mae_val = float(np.mean(np.abs(target_np - pred)))
            psnr_val = 10 * np.log10(1.0 / mse_val) if mse_val > 1e-10 else 100.0

            scores["ssim"].append(float(score_ssim))
            scores["mse"].append(mse_val)
            scores["mae"].append(mae_val)
            scores["psnr"].append(float(psnr_val))

            label = f"pred_frame_{idx:03d}"
            np.save(os.path.join(dirs["npy"], f"{label}.npy"), pred)
            save_image(pred, os.path.join(dirs["raw"], f"{label}.png"))
            save_heatmap(
                pred,
                os.path.join(dirs["raw_vis"], f"{label}.png"),
                use_gamma=True,
            )

            cdf_mapped = apply_quantile_mapping(pred)
            save_image(cdf_mapped, os.path.join(dirs["cdf"], f"{label}.png"))
            save_heatmap(
                cdf_mapped,
                os.path.join(dirs["cdf_vis"], f"{label}.png"),
                cmap_name="turbo",
                use_gamma=False,
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(target_np ** 0.3, cmap="magma", vmin=0, vmax=1)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            axes[1].imshow(pred ** 0.3, cmap="magma", vmin=0, vmax=1)
            axes[1].set_title(f"Prediction (SSIM: {score_ssim:.4f})")
            axes[1].axis("off")

            axes[2].imshow(np.abs(pred - target_np), cmap="hot", vmin=0, vmax=1)
            axes[2].set_title(f"Abs Diff (MSE: {mse_val:.4f})")
            axes[2].axis("off")

            plt.tight_layout(pad=2.0)
            plt.savefig(os.path.join(dirs["compare"], f"compare_{idx:03d}.png"))
            plt.close(fig)

    if scores["ssim"]:
        print(
            f"平均 SSIM: {np.mean(scores['ssim']):.4f} | "
            f"平均 PSNR: {np.mean(scores['psnr']):.4f}"
        )
        print(
            f"平均 MSE: {np.mean(scores['mse']):.6f} | "
            f"平均 MAE: {np.mean(scores['mae']):.6f}"
        )

        base_dir = os.path.dirname(output_dir)
        plt.figure(figsize=(12, 5))
        plt.plot(scores["ssim"], marker="o", linewidth=2, markersize=5)
        plt.axhline(
            y=np.mean(scores["ssim"]),
            color="red",
            linestyle="--",
            label=f"Mean SSIM: {np.mean(scores['ssim']):.4f}",
        )
        plt.xlabel("Frame Index")
        plt.ylabel("SSIM Score")
        plt.title("SSIM Score per Frame")
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "ssim_curve.png"), dpi=150)
        plt.close()
    else:
        print("测试集没有可评估的序列。")

    return scores


def collect_pairs(data_dir, cov_dir):
    """Pair input matrices by the shared timestamp filename, never by index."""
    main_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    cov_paths = sorted(glob.glob(os.path.join(cov_dir, "*.npy")))
    cov_map = {Path(path).stem: path for path in cov_paths}
    main_map = {Path(path).stem: path for path in main_paths}

    missing_cov = sorted(set(main_map) - set(cov_map))
    missing_energy = sorted(set(cov_map) - set(main_map))
    if missing_cov or missing_energy:
        raise ValueError(
            "能量场和 red_cov 文件名未完全对应。\n"
            f"缺少 red_cov 的标签（最多显示 10 个）: {missing_cov[:10]}\n"
            f"缺少能量场的标签（最多显示 10 个）: {missing_energy[:10]}"
        )

    pairs = [(main_map[key], cov_map[key]) for key in sorted(main_map)]
    print(
        f"找到 {len(main_paths)} 个能量场、{len(cov_paths)} 个 red_cov 场，"
        f"成功配对 {len(pairs)} 帧。"
    )
    return pairs


# ==================== One experiment per projection ====================
def run_experiment(wp, device):
    start_time = time.time()
    print("\n" + "=" * 60)
    print(f"开始视角 {wp} 的预测任务（red_cov 早期融合）")
    print("=" * 60)

    config = {
        "projection": wp,
        "seq_len": 10,
        "batch_size": 4,
        "epochs": 60,
        "hidden_dim": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
        "data_dir": str(PROJECT_ROOT / "input_txyz" / "e_data" / "matrix_npy" / wp),
        "cov_dir": str(PROJECT_ROOT / "input_txyz" / "red_cov" / "matrix_npy" / wp),
    }

    output_dir = PROJECT_ROOT / "output_txyz" / "red_cov" / f"{wp}_pred_npy_convlstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(output_dir)

    with open(
        os.path.join(output_dir, "hyperparameters.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    paired_paths = collect_pairs(config["data_dir"], config["cov_dir"])
    seq_len = config["seq_len"]
    if len(paired_paths) < seq_len + 2:
        raise ValueError(
            f"视角 {wp} 只有 {len(paired_paths)} 帧，至少需要 {seq_len + 2} 帧。"
        )

    split_idx = int(0.8 * len(paired_paths))
    if split_idx <= seq_len:
        raise ValueError("训练帧不足以生成训练序列，请减小 seq_len。")

    train_pairs = paired_paths[:split_idx]
    # Include the preceding seq_len frames as context for the first test target.
    test_pairs = paired_paths[split_idx - seq_len :]

    full_train_dataset = MemPairedSequenceDataset(
        train_pairs, seq_len, device=device
    )
    if len(full_train_dataset) < 2:
        raise ValueError("训练序列不足，无法划分训练集和验证集。")

    train_size = int(0.9 * len(full_train_dataset))
    train_size = max(1, min(train_size, len(full_train_dataset) - 1))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = MemPairedSequenceDataset(
        test_pairs, seq_len, device=device
    )
    if len(test_dataset) == 0:
        raise ValueError("测试序列为空；请检查帧数和训练/测试划分。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    model = EarlyFusionConvLSTMPredictor(
        input_channels=2,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    parameter_count = sum(param.numel() for param in model.parameters())
    print(f"早期融合模型参数量: {parameter_count:,}")

    train_losses, val_losses, best_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        device=device,
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
    )

    torch.save(model.state_dict(), os.path.join(output_dir, "last_epoch_model.pth"))

    with open(
        os.path.join(output_dir, "loss_log.csv"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write("epoch,train_loss,val_loss\n")
        for idx, (train_loss, val_loss) in enumerate(
            zip(train_losses, val_losses), start=1
        ):
            file.write(f"{idx},{train_loss:.6f},{val_loss:.6f}\n")

    if train_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

    print(f"\n评估测试集 {wp}...")
    metrics = evaluate(
        model,
        test_dataset,
        device,
        os.path.join(output_dir, "predictions"),
    )

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"

    with open(
        os.path.join(output_dir, "results_summary.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write("=============== red_cov 早期融合结果 ===============\n")
        file.write(f"投影视角: {wp}\n")
        file.write(f"模型参数量: {parameter_count:,}\n")
        file.write(f"测试集帧数: {len(metrics['ssim'])}\n")
        file.write(f"总运行时长: {time_str}\n")
        file.write(
            f"最佳模型 Epoch: {best_epoch} "
            f"(总设定 Epochs: {config['epochs']})\n"
        )
        if metrics["ssim"]:
            file.write(f"平均 SSIM: {np.mean(metrics['ssim']):.4f}\n")
            file.write(f"平均 PSNR: {np.mean(metrics['psnr']):.4f}\n")
            file.write(f"平均 MSE: {np.mean(metrics['mse']):.6f}\n")
            file.write(f"平均 MAE: {np.mean(metrics['mae']):.6f}\n")
        file.write("====================================================\n")

    print(f"视角 {wp} 完成，耗时 {time_str}。结果目录: {output_dir}")


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Match the current pred_txyz_cov1.py setting. Add xz/yz/tx/ty/tz to run them too.
    wp_list = ["xy"]

    for wp in wp_list:
        run_experiment(wp, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清理 GPU 显存。")


if __name__ == "__main__":
    main()
