import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ==================== 特效：复合锐化损失函数 (L1 + GDL) ====================
class SharpCompositeLoss(nn.Module):
    def __init__(self, l1_weight=1.0, gdl_weight=1.0, energy_factor=15.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.gdl_weight = gdl_weight
        self.energy_factor = energy_factor

    def forward(self, pred, target):
        # 1. 能量加权 L1 损失 (保持能量极值的定位准确度)
        base_l1 = torch.nn.functional.l1_loss(pred, target, reduction='none')
        weight_mask = 1.0 + target * self.energy_factor
        weighted_l1_loss = (base_l1 * weight_mask).mean()

        # 2. 梯度差异损失 GDL (惩罚边缘模糊，逼迫图像锐化)
        # 计算预测图和目标图在 X 和 Y 方向的空间梯度
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        gdl_loss = torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean()

        return self.l1_weight * weighted_l1_loss + self.gdl_weight * gdl_loss


# ==================== 特效：PyTorch可导 SSIM 模块 ====================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class DifferentiableSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(DifferentiableSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

# ==================== 特效：L1 + (1-SSIM) 混合损失函数 ====================
class L1_SSIM_MixedLoss(nn.Module):
    def __init__(self, alpha=0.5, energy_factor=15.0):
        super().__init__()
        self.alpha = alpha  # SSIM的权重比例 (0.5 表示 L1 和 SSIM 各占一半贡献)
        self.energy_factor = energy_factor
        self.ssim_module = DifferentiableSSIM(window_size=11)

    def forward(self, pred, target):
        # 1. 带有能量极值加权的 L1 损失 (惩罚微震点亮度不足)
        base_l1 = torch.nn.functional.l1_loss(pred, target, reduction='none')
        weight_mask = 1.0 + target * self.energy_factor
        weighted_l1_loss = (base_l1 * weight_mask).mean()

        # 2. 结构相似度度量 (惩罚形状畸变与模糊)
        ssim_val = self.ssim_module(pred, target)
        ssim_loss = 1.0 - ssim_val

        # 3. 按 alpha 比例混合
        return (1.0 - self.alpha) * weighted_l1_loss + self.alpha * ssim_loss