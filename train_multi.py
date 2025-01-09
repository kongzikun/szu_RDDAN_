import argparse
import os
import copy
import json

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim

def save_and_plot_metrics(outputs_dir, history):
    """保存并绘制训练指标"""
    # 保存训练历史到JSON
    history_path = os.path.join(outputs_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 绘制训练曲线
    epochs = list(range(1, len(history['train_losses']) + 1))
    
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_losses'], label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # PSNR曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['eval_psnrs'], label='PSNR', color='green')
    if 'best_metrics' in history:
        best_epoch = history['best_metrics']['epoch']
        best_psnr = history['best_metrics']['psnr']
        plt.scatter(best_epoch + 1, best_psnr, color='red', marker='*', s=100, 
                   label=f'Best: {best_psnr:.2f}dB')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Progress')
    plt.legend()
    plt.grid(True)
    
    # SSIM曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['eval_ssims'], label='SSIM', color='purple')
    if 'best_metrics' in history:
        plt.scatter(best_epoch + 1, history['best_metrics']['ssim'], color='red', 
                   marker='*', s=100, label=f'Best: {history["best_metrics"]["ssim"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train(args, model, criterion, optimizer, train_dataloader, device):
    """训练一个epoch"""
    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataloader.dataset) - len(train_dataloader.dataset) % args.batch_size)) as t:
        t.set_description(f'epoch: {args.current_epoch}/{args.num_epochs - 1}')

        for data in train_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)
            loss = criterion(preds, labels)
            
            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    return epoch_losses.avg

def evaluate(model, eval_dataloader, device):
    """评估模型"""
    model.eval()
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

    return epoch_psnr.avg, epoch_ssim.avg

def save_checkpoint(model, optimizer, epoch, history, outputs_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history
    }
    
    if is_best:
        path = os.path.join(outputs_dir, 'best.pth')
    else:
        path = os.path.join(outputs_dir, f'epoch_{epoch}.pth')
    
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    """加载检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', -1), checkpoint.get('history', {})

def main(args):
    """主训练函数"""
    # 设置设备和随机种子
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    # 保存训练配置
    config_path = os.path.join(args.outputs_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 初始化训练历史字典
    training_history = {
        'train_losses': [],
        'eval_psnrs': [],
        'eval_ssims': [],
        'best_metrics': {
            'psnr': 0.0,
            'ssim': 0.0,
            'epoch': 0
        }
    }

    # 初始化模型、损失函数和优化器
    model = SRCNN(
        num_channels=args.num_channels,
        num_features=args.num_features,
        growth_rate=args.growth_rate,
        num_blocks=args.num_blocks,
        num_layers=args.num_layers
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 如果提供了预训练模型，加载它
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            start_epoch, training_history = load_checkpoint(args.resume, model, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")

    # 初始化数据加载器
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)
    
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 记录最佳模型
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = training_history['best_metrics']['epoch']
    best_psnr = training_history['best_metrics']['psnr']
    best_ssim = training_history['best_metrics']['ssim']

    print(f"Starting training for {args.num_epochs} epochs...")
    try:
        # 主训练循环
        for epoch in range(start_epoch, args.num_epochs):
            args.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = train(args, model, criterion, optimizer, train_dataloader, device)
            training_history['train_losses'].append(train_loss)

            # 评估
            eval_psnr, eval_ssim = evaluate(model, eval_dataloader, device)
            training_history['eval_psnrs'].append(float(eval_psnr))
            training_history['eval_ssims'].append(float(eval_ssim))

            # 打印当前结果
            print(f'Epoch {epoch}: Loss: {train_loss:.6f}, PSNR: {eval_psnr:.2f}dB, SSIM: {eval_ssim:.4f}')

            # 更新最佳模型
            if eval_psnr > best_psnr:
                best_epoch = epoch
                best_psnr = eval_psnr
                best_ssim = eval_ssim
                best_weights = copy.deepcopy(model.state_dict())
                
                # 更新最佳指标记录
                training_history['best_metrics'] = {
                    'psnr': float(best_psnr),
                    'ssim': float(best_ssim),
                    'epoch': best_epoch
                }
                
                # 保存最佳模型
                save_checkpoint(model, optimizer, epoch, training_history, 
                              args.outputs_dir, is_best=True)
                print(f"Saved new best model with PSNR: {best_psnr:.2f}dB")

            # 定期保存检查点
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, epoch, training_history, args.outputs_dir)

            # 保存和绘制指标
            save_and_plot_metrics(args.outputs_dir, training_history)

    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nError during training: {str(e)}')
    finally:
        # 确保保存最终结果
        save_checkpoint(model, optimizer, epoch, training_history, args.outputs_dir)
        save_and_plot_metrics(args.outputs_dir, training_history)
        
        # 加载并保存最佳模型
        model.load_state_dict(best_weights)
        save_checkpoint(model, optimizer, best_epoch, training_history, 
                       args.outputs_dir, is_best=True)

    # 打印最终结果
    print('\nTraining completed.')
    print('Best epoch: {}, PSNR: {:.2f}dB, SSIM: {:.4f}'.format(
        best_epoch, best_psnr, best_ssim))
    
    return training_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True,
                        help='path to training dataset')
    parser.add_argument('--eval-file', type=str, required=True,
                        help='path to evaluation dataset')
    parser.add_argument('--outputs-dir', type=str, required=True,
                        help='path to save outputs')
    parser.add_argument('--scale', type=int, default=4,
                        help='super resolution scale')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--num-epochs', type=int, default=800,
                        help='number of epochs to train')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers for data loading')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed')
    parser.add_argument('--num-features', type=int, default=64,
                        help='number of features')
    parser.add_argument('--growth-rate', type=int, default=32,
                        help='growth rate')
    parser.add_argument('--num-blocks', type=int, default=3,
                        help='number of dense blocks')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='number of layers in each block')
    parser.add_argument('--num-channels', type=int, default=3,
                        help='number of input channels')
    parser.add_argument('--resume', type=str, default='',
                        help='path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=10,
                        help='save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # 在输出目录中创建尺度子目录
    args.outputs_dir = os.path.join(args.outputs_dir, f'x{args.scale}')
    
    main(args)