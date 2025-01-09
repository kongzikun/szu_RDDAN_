import matplotlib.pyplot as plt
import json
import os

def create_comparison_plots(all_results, param_counts, experiment_dir):
    """创建所有配置的对比图"""
    # 使用默认样式而不是seaborn
    plt.style.use('default')
    
    # 设置Colab兼容的字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建PSNR对比图
    plt.figure(figsize=(12, 6))
    for config_name, history in all_results.items():
        epochs = range(1, len(history['eval_psnrs']) + 1)
        plt.plot(epochs, history['eval_psnrs'], label=f'Config {config_name}', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison Across Different Configurations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'psnr_comparison.png'), 
                bbox_inches='tight', dpi=300)
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

    plt.scatter(param_counts_list, best_psnrs, s=100, c='blue', alpha=0.6)
    
    # 添加配置名称标签
    for i, config_name in enumerate(config_names):
        plt.annotate(config_name, 
                    (param_counts_list[i], best_psnrs[i]),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    plt.xlabel('Number of Parameters (100k)')
    plt.ylabel('Best PSNR (dB)')
    plt.title('Best PSNR vs Model Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'psnr_params_comparison.png'), 
                bbox_inches='tight', dpi=300)
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