import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from sklearn.preprocessing import QuantileTransformer

# ==================== 特效图像保存与处理工具 ====================
def save_image(img_array: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)

def save_heatmap(img_array: np.ndarray, path: str, cmap_name: str = "magma", use_gamma: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_clean = np.clip(img_array, 0, 1)
    
    if use_gamma:
        gamma = 0.3 
        img_clean = np.power(img_clean, gamma)
    
    cmap = plt.get_cmap(cmap_name)
    color_img = cmap(img_clean)
    img_uint8 = (color_img[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def apply_quantile_mapping(image_matrix: np.ndarray, background_threshold: float = 1e-5):
    H, W = image_matrix.shape
    flat_img = image_matrix.flatten().reshape(-1, 1)
    
    valid_mask = flat_img > background_threshold
    valid_pixels = flat_img[valid_mask].reshape(-1, 1)
    
    if len(valid_pixels) < 10:
        return np.zeros_like(image_matrix)
        
    n_quantiles = min(1000, len(valid_pixels))
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    
    mapped_valid = qt.fit_transform(valid_pixels)
    
    mapped_img_flat = np.zeros_like(flat_img, dtype=np.float32)
    mapped_img_flat[valid_mask] = mapped_valid.flatten()
    
    return mapped_img_flat.reshape(H, W)