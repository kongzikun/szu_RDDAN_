import json 
import os
from plot_results import create_comparison_plots
from visualize_comparison import visualize_results

def load_results():
    experiment_dir = 'experiments'
    all_results = {}
    param_counts = {}
    
    # 遍历所有配置目录
    configs = ['B1L1', 'B1L10', 'B3L3', 'B10L3', 'B20L3']
    
    # 创建配置列表用于可视化
    visual_configs = []
    
    for config in configs:
        config_dir = os.path.join(experiment_dir, config)
        if os.path.exists(config_dir):
            # 使用正确的文件名
            history_file = os.path.join(config_dir, 'training_history.json')
            print(f"检查配置 {config} 的历史文件: {history_file}")
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    try:
                        history_data = json.load(f)
                        print(f"成功加载 {config} 的历史数据")
                        # 确保数据格式正确
                        if isinstance(history_data, dict):
                            all_results[config] = history_data
                            # 添加到可视化配置
                            num_blocks = int(config[1:config.index('L')])
                            num_layers = int(config[config.index('L')+1:])
                            visual_configs.append({
                                'name': config,
                                'num_blocks': num_blocks,
                                'num_layers': num_layers
                            })
                        else:
                            print(f"警告: {config} 的数据格式不正确")
                    except json.JSONDecodeError as e:
                        print(f"错误: 无法解析 {config} 的JSON数据: {e}")
            else:
                print(f"警告: 找不到 {config} 的历史文件")
            
            # 计算参数量
            num_features = 64
            growth_rate = 32
            num_blocks = int(config[1:config.index('L')])
            num_layers = int(config[config.index('L')+1:])
            
            param_count = calculate_model_params(
                num_blocks,
                num_layers,
                num_features,
                growth_rate
            )
            param_counts[config] = param_count
    
    return all_results, param_counts, visual_configs

def calculate_model_params(num_blocks, num_layers, num_features, growth_rate):
    total_params = 0
    current_features = num_features
    
    for _ in range(num_blocks):
        for _ in range(num_layers):
            layer_params = (current_features * 3 * 3 + 1) * growth_rate
            total_params += layer_params
            current_features += growth_rate
        
        if _ < num_blocks - 1:
            transition_params = (current_features * 1 * 1 + 1) * (current_features // 2)
            total_params += transition_params
            current_features = current_features // 2
    
    final_params = (current_features * 3 * 3 + 1) * (3 * 16)
    total_params += final_params
    
    return total_params

if __name__ == '__main__':
    # 加载结果
    all_results, param_counts, visual_configs = load_results()
    
    if not all_results:
        print("错误: 没有成功加载任何结果数据")
    else:
        print(f"成功加载了 {len(all_results)} 个配置的数据")
        # 创建训练曲线图表
        create_comparison_plots(all_results, param_counts, 'experiments')
        
        # 设置测试图像路径
        test_image_path = '/content/drive/MyDrive/SRCNN/dataset/Set5-test/baby.png'
        
        # 检查测试图像是否存在
        if os.path.exists(test_image_path):
            print(f"\n开始生成可视化对比...")
            print(f"使用测试图像: {test_image_path}")
            try:
                visualize_results(test_image_path, visual_configs, 'experiments')
                print("可视化对比完成！")
            except Exception as e:
                print(f"可视化过程中出现错误: {e}")
                import traceback
                traceback.print_exc()  # 打印详细的错误信息
        else:
            print(f"\n警告: 找不到测试图像 {test_image_path}")
            print("请确保已经挂载了Google Drive并且路径正确")