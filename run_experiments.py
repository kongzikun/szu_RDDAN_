import os
import json
import argparse
import matplotlib.pyplot as plt
from train import main as train_main
from argparse import Namespace

def calculate_model_params(num_blocks, num_layers, num_features, growth_rate):
    """计算模型参数量
    
    Args:
        num_blocks (int): 块的数量
        num_layers (int): 每个块中的层数
        num_features (int): 初始特征数
        growth_rate (int): 特征增长率
    
    Returns:
        int: 模型总参数量
    """
    total_params = 0
    current_features = num_features
    
    for _ in range(num_blocks):
        for _ in range(num_layers):
            # 每层的参数量：conv层的权重和偏置
            # 3x3 卷积的参数量: (in_channels * 3 * 3 + 1) * out_channels
            layer_params = (current_features * 3 * 3 + 1) * growth_rate
            total_params += layer_params
            current_features += growth_rate
        
        # 转换层的参数量（如果不是最后一个块）
        if _ < num_blocks - 1:
            transition_params = (current_features * 1 * 1 + 1) * (current_features // 2)
            total_params += transition_params
            current_features = current_features // 2
    
    # 最后的重建层参数量
    final_params = (current_features * 3 * 3 + 1) * (3 * 16)  # 假设scale=4，输出通道为3
    total_params += final_params
    
    return total_params

def run_all_configurations():
    # 基础配置参数
    base_args = {
        'train_file': 'dataset/train_data.h5',
        'eval_file': 'dataset/test_data.h5',
        'scale': 4,
        'lr': 1e-4,
        'batch_size': 16,
        'num_epochs': 200,
        'num_workers': 8,
        'seed': 123,
        'num_features': 64,
        'growth_rate': 32
    }

    # 定义要测试的不同配置
    configurations = [
        {'num_blocks': 1, 'num_layers': 1, 'name': 'B1L1'},
        {'num_blocks': 1, 'num_layers': 10, 'name': 'B1L10'},
        {'num_blocks': 3, 'num_layers': 3, 'name': 'B3L3'},
        {'num_blocks': 10, 'num_layers': 3, 'name': 'B10L3'},
        {'num_blocks': 20, 'num_layers': 3, 'name': 'B20L3'}
    ]

    # 创建主实验目录
    experiment_dir = 'experiments'
    os.makedirs(experiment_dir, exist_ok=True)

    # 存储所有配置的结果
    all_results = {}
    param_counts = {}

    # 运行每个配置
    for config in configurations:
        print(f"\n开始训练配置: {config['name']}")
        print(f"Blocks: {config['num_blocks']}, Layers: {config['num_layers']}")

        # 计算参数量
        param_count = calculate_model_params(
            config['num_blocks'],
            config['num_layers'],
            base_args['num_features'],
            base_args['growth_rate']
        )
        param_counts[config['name']] = param_count
        
        # 创建此配置的输出目录
        output_dir = os.path.join(experiment_dir, config['name'])
        
        # 准备参数
        current_args = base_args.copy()
        current_args.update(config)
        current_args['outputs_dir'] = output_dir
        
        # 转换为Namespace对象
        args = Namespace(**current_args)
        
        # 运行训练
        history = train_main(args)
        all_results[config['name']] = history

    # 创建对比图
    create_comparison_plots(all_results, param_counts, experiment_dir)

def create_comparison_plots(all_results, param_counts, experiment_dir):
    """创建所有配置的对比图"""
    # 设置图表风格
    plt.style.use('seaborn')
    
    # 创建PSNR对比图
    plt.figure(figsize=(12, 6))
    for config_name, history in all_results.items():
        epochs = range(1, len(history['eval_psnrs']) + 1)
        plt.plot(epochs, history['eval_psnrs'], label=f'Config {config_name}')
    
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison Across Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'psnr_comparison.png'))
    plt.close()

    # 创建PSNR-参数量对比图
    plt.figure(figsize=(10, 6))
    param_counts_list = []
    best_psnrs = []
    config_names = []
    
    for config_name, history in all_results.items():
        param_counts_list.append(param_counts[config_name] / 1e5)  # 转换为10万
        best_psnrs.append(history['best_metrics']['psnr'])
        config_names.append(config_name)

    plt.scatter(param_counts_list, best_psnrs, s=100)
    
    # 添加配置名称标签
    for i, config_name in enumerate(config_names):
        plt.annotate(config_name, 
                    (param_counts_list[i], best_psnrs[i]),
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.xlabel('Number of Parameters (100k)')
    plt.ylabel('Best PSNR (dB)')
    plt.title('Best PSNR vs Model Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'psnr_params_comparison.png'))
    plt.close()

    # 创建最佳结果摘要
    summary = {
        "Best Results": {},
        "Model Sizes": {}
    }
    
    for config_name, history in all_results.items():
        summary["Best Results"][config_name] = {
            "Best PSNR": history['best_metrics']['psnr'],
            "Best SSIM": history['best_metrics']['ssim'],
            "Best Epoch": history['best_metrics']['epoch']
        }
        summary["Model Sizes"][config_name] = {
            "Parameters (100k)": param_counts[config_name] / 1e5
        }

    # 保存结果摘要
    with open(os.path.join(experiment_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # 打印结果摘要
    print("\n=== 实验结果摘要 ===")
    for config_name, results in summary["Best Results"].items():
        print(f"\nConfiguration: {config_name}")
        print(f"Parameters: {summary['Model Sizes'][config_name]['Parameters (100k)']:.2f}x100k")
        print(f"Best PSNR: {results['Best PSNR']:.2f} dB")
        print(f"Best SSIM: {results['Best SSIM']:.4f}")
        print(f"Achieved at epoch: {results['Best Epoch']}")

if __name__ == '__main__':
    run_all_configurations()