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
    # 确保张量在CPU上且为numpy数组
    image = tensor.cpu().numpy().transpose((1, 2, 0))
    # 将值范围从[0,1]转换到[0,255]
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    # 如果是单通道图像，去掉通道维度
    if image.shape[2] == 1:
        image = image[:, :, 0]
    # 保存图像
    Image.fromarray(image).save(filename)

def load_model(model_path, device):
    """加载预训练模型"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从检查点获取模型配置
    model_config = checkpoint['model_config']
    
    # 创建模型实例
    model = SRCNN_RDN_DCN_ECA(
        num_channels=model_config['num_channels'],
        num_features=model_config['num_features'],
        growth_rate=model_config['growth_rate'],
        num_blocks=model_config['num_blocks'],
        num_layers=model_config['num_layers']
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['state_dict'])
    return model

def test(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.model_path, device)
    model.eval()
    
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
                save_image(
                    prediction[0],
                    os.path.join(args.output_dir, f'output_{idx:04d}.png')
                )
                # 可选：保存输入和目标图像用于对比
                if args.save_comparison:
                    save_image(
                        input_tensor[0],
                        os.path.join(args.output_dir, f'input_{idx:04d}.png')
                    )
                    save_image(
                        target[0],
                        os.path.join(args.output_dir, f'target_{idx:04d}.png')
                    )
    
    # 打印最终结果
    print(f'Final Results:')
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
                        help='Save input and target images along with output')
    
    args = parser.parse_args()
    
    test(args)