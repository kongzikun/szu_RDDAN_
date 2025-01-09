import os
import json
import matplotlib.pyplot as plt
import argparse

def load_training_history(json_path):
    """加载训练历史JSON文件"""
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    losses = history['losses']
    psnrs = history['psnrs']
    epochs = range(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(121)
    plt.plot(epochs, losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制PSNR曲线
    plt.subplot(122)
    plt.plot(epochs, psnrs, 'r-')
    plt.title('PSNR on Validation Set')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'已保存训练曲线到: {save_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='weight/x4/training_history.json',
                      help='训练历史JSON文件路径')
    parser.add_argument('--save-plot', type=str, default='Result/training_curves.png',
                      help='保存图像的路径')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    
    # 加载训练历史
    history = load_training_history(args.history)
    
    # 绘制并保存训练曲线
    plot_training_curves(history, args.save_plot)
