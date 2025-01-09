import os
import json
import matplotlib.pyplot as plt
import argparse

def plot_training_curves(model_dirs, save_path='model_comparison.png', metrics=['psnr', 'ssim']):
    """
    绘制多个模型的训练曲线对比图
    
    Args:
        model_dirs: 模型结果目录的列表，每个目录应包含training_history.json
        save_path: 保存图像的路径
        metrics: 要绘制的指标列表，可以是'psnr'和'ssim'
    """
    plt.figure(figsize=(12, 5 * len(metrics)))
    
    for idx, metric in enumerate(metrics, 1):
        plt.subplot(len(metrics), 1, idx)
        
        for model_dir in model_dirs:
            try:
                # 读取训练历史
                history_path = os.path.join(model_dir, 'training_history.json')
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                # 获取模型名称（使用目录的最后一部分）
                model_name = os.path.basename(os.path.normpath(model_dir))
                
                # 绘制曲线
                metric_key = f'eval_{metric}s'  # 对应JSON中的键名
                if metric_key in history:
                    epochs = range(1, len(history[metric_key]) + 1)
                    plt.plot(epochs, history[metric_key], label=f'{model_name}')
                    
                    # 标注最佳值
                    best_value = max(history[metric_key])
                    best_epoch = history[metric_key].index(best_value) + 1
                    plt.scatter(best_epoch, best_value, marker='*', s=100)
                    plt.annotate(f'Best: {best_value:.2f}', 
                               (best_epoch, best_value),
                               xytext=(10, 10), 
                               textcoords='offset points')
                
            except Exception as e:
                print(f"处理 {model_dir} 时出错: {str(e)}")
        
        plt.title(f'{metric.upper()} Comparison')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"图像已保存到: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics comparison')
    parser.add_argument('--model-dirs', nargs='+', required=True,
                      help='模型结果目录的路径列表')
    parser.add_argument('--save-path', default='model_comparison.png',
                      help='保存图像的路径')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim'],
                      help='要绘制的指标列表')
    
    args = parser.parse_args()
    
    plot_training_curves(args.model_dirs, args.save_path, args.metrics)
