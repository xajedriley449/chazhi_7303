import os
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset, Subset


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "input_txyz").is_dir():
            return candidate
    raise FileNotFoundError(
        "无法定位项目根目录：请将脚本放在项目目录或其子目录中，"
        "并确认项目根目录下存在 input_txyz/。"
    )


PROJECT_ROOT = find_project_root()
PROJECTIONS = ["xy"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"找不到矩阵文件：{path}")
    arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"预期二维矩阵，实际形状为 {arr.shape}：{path}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_energy_frame(arr: np.ndarray) -> np.ndarray:
    """沿用 red_cov 训练脚本的能量场逐帧 [0, 1] 归一化。"""
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)


def collect_pairs(data_dir: Path, cov_dir: Path) -> list[tuple[Path, Path]]:
    """按时间戳文件名配对，不依赖目录枚举顺序。"""
    main_map = {path.stem: path for path in data_dir.glob("*.npy")}
    cov_map = {path.stem: path for path in cov_dir.glob("*.npy")}
    if not main_map:
        raise FileNotFoundError(f"能量场目录没有 .npy 文件：{data_dir}")
    missing_cov = sorted(set(main_map) - set(cov_map))
    missing_energy = sorted(set(cov_map) - set(main_map))
    if missing_cov or missing_energy:
        raise ValueError(
            "能量场与 coal_level 文件名未完全对应。\n"
            f"缺少 coal_level 的标签（最多显示 10 个）：{missing_cov[:10]}\n"
            f"缺少能量场的标签（最多显示 10 个）：{missing_energy[:10]}"
        )

    pairs = [(main_map[key], cov_map[key]) for key in sorted(main_map)]
    print(f"找到 {len(pairs)} 组按时间戳配对的能量场/煤层场。")
    return pairs


def get_training_cov_scale(
    train_pairs: list[tuple[Path, Path]],
) -> float:
    """用训练时间段的 coal_level 统一幅值，保持场的正负号和跨帧幅值关系。"""
    max_abs = 0.0
    for _, cov_path in train_pairs:
        arr = load_matrix(cov_path)
        max_abs = max(max_abs, float(np.max(np.abs(arr))))
    return max(max_abs, 1e-8)


class PairedSequenceDataset(Dataset):
    def __init__(
        self,
        paired_paths: list[tuple[Path, Path]],
        seq_len: int,
        cov_scale: float,
    ) -> None:
        if len(paired_paths) <= seq_len:
            raise ValueError(
                f"帧数 {len(paired_paths)} 必须大于 seq_len={seq_len}。"
            )

        self.seq_len = seq_len
        self.labels = [main_path.stem for main_path, _ in paired_paths]
        main_frames = []
        cov_frames = []
        expected_shape = None

        print("正在读取能量场和 coal_level 场...")
        for main_path, cov_path in paired_paths:
            main_arr = normalize_energy_frame(load_matrix(main_path))
            cov_arr = load_matrix(cov_path)
            if main_arr.shape != cov_arr.shape:
                raise ValueError(
                    "能量场与煤层场矩阵尺寸不一致：\n"
                    f"  energy: {main_path} {main_arr.shape}\n"
                    f"  coal_level: {cov_path} {cov_arr.shape}"
                )
            if expected_shape is None:
                expected_shape = main_arr.shape
            elif main_arr.shape != expected_shape:
                raise ValueError(
                    f"矩阵尺寸不统一：{main_path} {main_arr.shape}，"
                    f"预期 {expected_shape}。"
                )

            # 对所有帧使用同一训练段尺度，不逐帧归一化，避免抹掉符号与幅值变化。
            cov_arr = np.clip(cov_arr / cov_scale, -1.0, 1.0).astype(np.float32)
            main_frames.append(torch.from_numpy(main_arr).unsqueeze(0))
            cov_frames.append(torch.from_numpy(cov_arr).unsqueeze(0))

        self.all_main = torch.stack(main_frames, dim=0)
        self.all_cov = torch.stack(cov_frames, dim=0)
        print(
            f"已加载 {len(self.all_main)} 帧，矩阵尺寸 {tuple(self.all_main.shape[-2:])}；"
            f"coal_level 训练段 scale={cov_scale:.6g}。"
        )

    def __len__(self) -> int:
        return len(self.all_main) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_energy = self.all_main[idx : idx + self.seq_len]
        seq_coal = self.all_cov[idx : idx + self.seq_len]
        x_seq = torch.cat([seq_energy, seq_coal], dim=1)
        target = self.all_main[idx + self.seq_len]
        return x_seq, target


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        gates = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.hidden_dim, height, width)
        return reference.new_zeros(shape), reference.new_zeros(shape)


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_dim=input_dim if layer_idx == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # x: [batch, time, channels, height, width]
        batch_size, seq_len, _, height, width = x.shape
        layer_input = x
        last_states = []

        for cell in self.cells:
            h, c = cell.init_hidden(
                batch_size, height, width, layer_input
            )
            outputs = []
            for time_idx in range(seq_len):
                h, c = cell(layer_input[:, time_idx], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)
            last_states.append((h, c))
        return last_states


class EarlyFusionConvLSTMPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.convlstm = ConvLSTM(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=3,
        )
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_states = self.convlstm(x)
        final_hidden = last_states[-1][0]
        return torch.sigmoid(self.output_conv(final_hidden))


class SharpCompositeLoss(nn.Module):
    """与项目 red_cov 训练使用的加权 L1 + 梯度差异损失一致。"""
    def __init__(
        self,
        l1_weight: float = 1.0,
        gdl_weight: float = 5.0,
        energy_factor: float = 15.0,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.gdl_weight = gdl_weight
        self.energy_factor = energy_factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_l1 = torch.abs(pred - target)
        weight_mask = 1.0 + target * self.energy_factor
        weighted_l1 = (base_l1 * weight_mask).mean()

        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        gdl = torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean()
        return self.l1_weight * weighted_l1 + self.gdl_weight * gdl


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    output_dir: Path,
    learning_rate: float,
    early_stop_patience: int = 15,
) -> tuple[list[float], list[float], int]:
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("训练或验证 DataLoader 为空，请检查帧数、seq_len 和 batch_size。")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = SharpCompositeLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_path = output_dir / "best_model.pth"
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    saved_best_this_run = False

    print("训练过程中可按 Ctrl+C 中断；结束后使用本次训练的最佳验证模型。")
    try:
        for epoch in range(epochs):
            model.train()
            train_sum = 0.0
            for seq, target in train_loader:
                seq = seq.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(seq), target)
                loss.backward()
                optimizer.step()
                train_sum += float(loss.item())

            train_loss = train_sum / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for seq, target in val_loader:
                    seq = seq.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    val_sum += float(criterion(model(seq), target).item())
            val_loss = val_sum / len(val_loader)
            val_losses.append(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{epochs}] LR={current_lr:.6g} | "
                f"Train={train_loss:.6f} | Val={val_loss:.6f}"
            )
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                saved_best_this_run = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stop_patience:
                print(f"验证损失连续 {early_stop_patience} 个 epoch 未改善，提前停止。")
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，结束训练并整理已有结果。")

    if saved_best_this_run:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"已加载最佳验证模型，epoch={best_epoch}，val_loss={best_val_loss:.6f}。")
    else:
        # 若在第一个验证 epoch 前中断，则保存当前模型，避免误用旧运行的检查点。
        torch.save(model.state_dict(), best_path)
        print("本次训练尚无验证检查点，已保存当前模型。")
    return train_losses, val_losses, best_epoch


def evaluate(
    model: nn.Module,
    dataset: PairedSequenceDataset,
    test_indices: range,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, float | str]]:
    model.eval()
    dirs = {
        "npy": output_dir / "matrix_npy",
        "vis": output_dir / "pred_vis",
        "compare": output_dir / "pred_images",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        for sample_idx in test_indices:
            seq, target = dataset[sample_idx]
            pred = model(seq.unsqueeze(0).to(device))[0, 0].cpu().numpy()
            target_np = target[0].cpu().numpy()
            label = dataset.labels[sample_idx + dataset.seq_len]

            score_ssim = float(ssim(target_np, pred, data_range=1.0))
            mse_val = float(np.mean((target_np - pred) ** 2))
            mae_val = float(np.mean(np.abs(target_np - pred)))
            psnr_val = float(10.0 * np.log10(1.0 / mse_val)) if mse_val > 1e-10 else 100.0
            rows.append(
                {
                    "label": label,
                    "ssim": score_ssim,
                    "mse": mse_val,
                    "mae": mae_val,
                    "psnr": psnr_val,
                }
            )

            np.save(dirs["npy"] / f"{label}.npy", pred.astype(np.float32))
            plt.imsave(dirs["vis"] / f"{label}.png", pred, cmap="magma", vmin=0, vmax=1)

            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            axes[0].imshow(target_np, cmap="magma", vmin=0, vmax=1)
            axes[0].set_title("Ground truth")
            axes[1].imshow(pred, cmap="magma", vmin=0, vmax=1)
            axes[1].set_title(f"Prediction, SSIM={score_ssim:.4f}")
            axes[2].imshow(np.abs(pred - target_np), cmap="hot", vmin=0, vmax=1)
            axes[2].set_title(f"Absolute error, MAE={mae_val:.4f}")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(dirs["compare"] / f"{label}.png", dpi=140)
            plt.close(fig)

    if rows:
        with (output_dir / "scores.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["label", "ssim", "mse", "mae", "psnr"])
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"测试集 {len(rows)} 帧：SSIM={np.mean([r['ssim'] for r in rows]):.4f}, "
            f"PSNR={np.mean([r['psnr'] for r in rows]):.4f}, "
            f"MSE={np.mean([r['mse'] for r in rows]):.6f}, "
            f"MAE={np.mean([r['mae'] for r in rows]):.6f}"
        )
    else:
        print("测试集没有可评估的序列。")
    return rows


def run_experiment(projection: str, device: torch.device) -> None:
    started_at = time.time()
    config = {
        "projection": projection,
        "seq_len": 10,
        "batch_size": 4,
        "epochs": 60,
        "hidden_dim": 32,
        "num_layers": 2,
        "learning_rate": 1e-3,
        "train_fraction": 0.8,
        "validation_fraction_within_train": 0.1,
        "seed": 42,
        "data_dir": str(PROJECT_ROOT / "input_txyz" / "e_data" / "matrix_npy" / projection),
        "cov_dir": str(PROJECT_ROOT / "input_txyz" / "coal_level" / "matrix_npy" / projection),
        "cov_normalization": "training-segment shared max-absolute scale; preserve sign",
    }
    output_dir = PROJECT_ROOT / "output_txyz" / "coal_level" / f"{projection}_pred_npy_convlstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hyperparameters.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"开始 {projection} 视角：energy + coal_level 早期融合")
    print(f"设备：{device}")

    paired_paths = collect_pairs(Path(config["data_dir"]), Path(config["cov_dir"]))
    seq_len = config["seq_len"]
    split_idx = int(len(paired_paths) * config["train_fraction"])
    if split_idx <= seq_len + 1 or len(paired_paths) - split_idx < 1:
        raise ValueError(
            f"帧数不足以按 seq_len={seq_len} 划分训练/测试：总帧数={len(paired_paths)}，"
            f"训练段帧数={split_idx}。"
        )

    cov_scale = get_training_cov_scale(paired_paths[:split_idx])
    dataset = PairedSequenceDataset(paired_paths, seq_len, cov_scale)

    # 样本索引代表预测目标所在帧减去 seq_len；验证目标位于训练段尾部。
    train_segment_samples = split_idx - seq_len
    val_size = max(1, int(train_segment_samples * config["validation_fraction_within_train"]))
    train_size = train_segment_samples - val_size
    if train_size < 1:
        raise ValueError("训练序列不足，请减小 seq_len 或调整划分比例。")
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_segment_samples)
    test_indices = range(split_idx - seq_len, len(paired_paths) - seq_len)

    train_loader = DataLoader(
        Subset(dataset, list(train_indices)),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, list(val_indices)),
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = EarlyFusionConvLSTMPredictor(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    param_count = sum(param.numel() for param in model.parameters())
    print(
        f"配对帧={len(paired_paths)}，训练样本={len(train_indices)}，"
        f"验证样本={len(val_indices)}，测试样本={len(test_indices)}，"
        f"模型参数={param_count:,}。"
    )

    train_losses, val_losses, best_epoch = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["epochs"],
        device=device,
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
    )

    with (output_dir / "loss_log.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(
            (idx, train_loss, val_loss)
            for idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1)
        )

    if train_losses:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
        ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss.png", dpi=140)
        plt.close(fig)

    metrics = evaluate(
        model=model,
        dataset=dataset,
        test_indices=test_indices,
        device=device,
        output_dir=output_dir / "predictions",
    )
    elapsed = time.time() - started_at
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"

    with (output_dir / "results_summary.txt").open("w", encoding="utf-8") as file:
        file.write("=============== coal_level 早期融合结果 ===============\n")
        file.write(f"投影视角: {projection}\n")
        file.write(f"模型参数量: {param_count:,}\n")
        file.write(f"配对总帧数: {len(paired_paths)}\n")
        file.write(f"训练样本数: {len(train_indices)}\n")
        file.write(f"验证样本数: {len(val_indices)}\n")
        file.write(f"测试帧数: {len(metrics)}\n")
        file.write(f"煤层场统一缩放值: {cov_scale:.8g}\n")
        file.write(f"最佳模型 Epoch: {best_epoch}\n")
        file.write(f"总运行时长: {time_str}\n")

        if metrics:
            for key in ("ssim", "psnr", "mse", "mae"):
                file.write(f"平均 {key.upper()}: {np.mean([row[key] for row in metrics]):.6f}\n")
        file.write("========================================================\n")

    print(f"\n{projection} 视角处理完成！耗时: {time_str}。结果保存在: {output_dir}")


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"Device: {device}")
    for projection in PROJECTIONS:
        run_experiment(projection, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
