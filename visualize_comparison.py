
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from math import exp
from models import SRCNN_RDN_DCN_ECA  # 假设模型文件名为model.py

def calc_psnr(img1, img2):
    """计算PSNR值"""
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def create_window(window_size, channel=1):
    """创建高斯窗口"""
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([
            exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _1d_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def calc_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    """计算SSIM值"""
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return ret

def visualize_results(test_image_path, configs, experiment_dir):
    """可视化不同配置模型的重建效果
    
    Args:
        test_image_path: 测试图像路径
        configs: 不同的模型配置列表
        experiment_dir: 实验结果保存目录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置图表样式
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    # 加载并预处理测试图像
    hr_image = Image.open(test_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 创建低分辨率图像
    hr_tensor = transform(hr_image).unsqueeze(0).to(device)
    lr_size = (hr_image.size[0] // 4, hr_image.size[1] // 4)
    lr_image = hr_image.resize(lr_size, Image.BICUBIC)
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)

    # 创建图表布局
    fig = plt.figure(figsize=(15, 10))
    
    # 显示原始图像和选定区域
    ax_main = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
    ax_main.imshow(np.array(hr_image))
    ax_main.set_title('Original HR Image')
    ax_main.axis('off')
    
    # 在原图上标记感兴趣区域
    rect = plt.Rectangle((100, 100), 50, 50, fill=False, color='red', linewidth=2)
    ax_main.add_patch(rect)

    # 为每个配置创建结果展示
    axes = [plt.subplot2grid((2, 4), (i//2, 2+i%2)) for i in range(4)]
    
    for ax, config in zip(axes, configs):
        # 加载对应配置的模型
        model = SRCNN_RDN_DCN_ECA(num_blocks=config['num_blocks'], 
                      num_layers=config['num_layers'],
                      num_features=64,
                      growth_rate=32)
        
        model_path = os.path.join(experiment_dir, config['name'], 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            # 生成超分辨率图像
            with torch.no_grad():
                sr_tensor = model(lr_tensor)
            
            # 计算PSNR和SSIM
            psnr = calc_psnr(sr_tensor, hr_tensor).item()
            ssim = calc_ssim(sr_tensor, hr_tensor).item()
            
            # 后处理
            sr_tensor = sr_tensor.cpu()
            sr_image = sr_tensor.squeeze().numpy()
            sr_image = (sr_image + 1) * 0.5  # 反归一化
            sr_image = np.clip(sr_image, 0, 1)
            sr_image = np.transpose(sr_image, (1, 2, 0))  # CHW -> HWW
            
            # 显示重建结果（仅显示感兴趣区域）
            roi = sr_image[100:150, 100:150]  # 假设这是感兴趣区域的坐标
            ax.imshow(roi)
            ax.set_title(f"{config['name']}\n{psnr:.2f}/{ssim:.4f}")
        else:
            print(f"Warning: Model not found for config {config['name']}")
        
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'visual_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    # 测试配置
    configs = [
        {'name': 'B1L1', 'num_blocks': 1, 'num_layers': 1},
        {'name': 'B1L10', 'num_blocks': 1, 'num_layers': 10},
        {'name': 'B3L3', 'num_blocks': 3, 'num_layers': 3},
        {'name': 'B10L3', 'num_blocks': 10, 'num_layers': 3}
    ]
    
    test_image_path = 'dataset/Set5-test/baby.png'  # 替换为实际的测试图像路径
    visualize_results(test_image_path, configs, 'experiments')
