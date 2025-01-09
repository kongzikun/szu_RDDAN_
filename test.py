import argparse
import os
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from models import SRCNN_RDN_DCN_ECA
from datasets import EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim

def save_image(tensor, filename):
    """将张量转换为图像并保存"""
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(filename)

def create_comparison_image(lr_tensor, sr_tensor, hr_tensor, save_path):
    """创建对比图像，将LR、SR和HR图像水平拼接，保持原始尺寸"""
    # 将张量转换为PIL图像
    def tensor_to_pil(tensor):
        image = tensor.cpu().numpy().transpose((1, 2, 0))
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    lr_img = tensor_to_pil(lr_tensor)
    sr_img = tensor_to_pil(sr_tensor)
    hr_img = tensor_to_pil(hr_tensor)
    
    # 创建新的空白图像用于拼接
    total_width = lr_img.width + sr_img.width + hr_img.width
    max_height = max(lr_img.height, sr_img.height, hr_img.height)
    
    # 创建带标签的对比图像
    margin = 20  # 标签的边距
    label_height = 30  # 标签的高度
    comparison = Image.new('RGB', (total_width, max_height + label_height), 'white')
    
    # 粘贴图像
    x_offset = 0
    for img, label in [(lr_img, 'LR'), (sr_img, 'SR'), (hr_img, 'HR')]:
        # 粘贴图像
        comparison.paste(img, (x_offset, label_height))
        
        # 在图像上方添加标签
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        # 计算文本位置使其居中
        label_width = draw.textlength(label)
        label_x = x_offset + (img.width - label_width) // 2
        draw.text((label_x, 5), label, fill='black')
        
        x_offset += img.width
    
    # 保存对比图像
    comparison.save(save_path)

def load_model(model_path, device):
    """加载预训练模型，使用保存的配置"""
    print(f'Loading model from {model_path}')
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从检查点获取模型配置
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # 分析state_dict来确定模型配置
            num_blocks = sum(1 for k in state_dict.keys() if k.startswith('rdbs.') and k.endswith('.layers.0.conv.weight'))
            print(f"Detected num_blocks: {num_blocks}")
            
            # 检测输入通道数
            first_layer_weight = state_dict['sfe1.weight']
            num_channels = first_layer_weight.size(1)
            print(f"Detected num_channels: {num_channels}")
        else:
            print("Warning: Could not find state_dict in checkpoint, using default configuration")
            num_blocks = 1
            num_channels = 3  # 默认使用3通道
        
        # 创建模型实例
        model = SRCNN_RDN_DCN_ECA(
            num_channels=num_channels,  # 使用检测到的通道数
            num_features=64,
            growth_rate=32,
            num_blocks=num_blocks,
            num_layers=3
        ).to(device)
        
        # 加载权重
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded model state from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded state dict directly")
        
        return model
        
    except Exception as e:
        print(f'Error loading model: {str(e)}')
        raise

def test(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_comparison:
        comparison_dir = os.path.join(args.output_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, device)
    model.eval()
    
    # 打印模型信息
    print("\nModel configuration:")
    if hasattr(model, 'get_model_info'):
        print(model.get_model_info())
    
    # 准备数据加载器
    test_dataset = EvalDataset(args.test_file)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)
    
    # 初始化评估指标
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    # 开始测试
    print('Starting evaluation...')
    with torch.no_grad():
        for idx, (input_tensor, target) in enumerate(tqdm(test_dataloader)):
            input_tensor = input_tensor.to(device)
            target = target.to(device)
            
            # 模型推理
            prediction = model(input_tensor).clamp(0.0, 1.0)
            
            # 计算评估指标
            psnr = calc_psnr(prediction, target)
            ssim = calc_ssim(prediction, target)
            
            psnr_meter.update(psnr, len(input_tensor))
            ssim_meter.update(ssim, len(input_tensor))
            
            # 保存结果图像
            if args.save_results:
                # 保存单独的图像
                save_image(
                    prediction[0],
                    os.path.join(args.output_dir, f'output_{idx:04d}.png')
                )
                if args.save_comparison:
                    # 保存对比图像
                    comparison_path = os.path.join(comparison_dir, f'comparison_{idx:04d}.png')
                    create_comparison_image(
                        input_tensor[0],    # LR
                        prediction[0],      # SR (output)
                        target[0],          # HR
                        comparison_path
                    )
    
    # 打印最终结果
    print(f'\nFinal Results:')
    print(f'Average PSNR: {psnr_meter.avg:.2f} dB')
    print(f'Average SSIM: {ssim_meter.avg:.4f}')
    
    # 保存评估结果到文本文件
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write(f'Model: {args.model_path}\n')
        f.write(f'Test dataset: {args.test_file}\n')
        f.write(f'Average PSNR: {psnr_meter.avg:.2f} dB\n')
        f.write(f'Average SSIM: {ssim_meter.avg:.4f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Directory to save test results')
    parser.add_argument('--save-results', action='store_true',
                        help='Save output images')
    parser.add_argument('--save-comparison', action='store_true',
                        help='Save comparison images with LR, SR, and HR side by side')
    
    args = parser.parse_args()
    
    test(args)