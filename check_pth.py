import torch
import os

def analyze_tensor(tensor):
    """分析张量的基本信息"""
    return {
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'numel': tensor.numel(),
        'memory_mb': tensor.numel() * tensor.element_size() / (1024*1024)
    }

def analyze_state_dict_summary(state_dict):
    """分析state_dict的摘要信息"""
    total_params = 0
    module_stats = {}
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 提取模块名（去掉.weight/.bias后缀）
            module_name = key.replace('.weight', '').replace('.bias', '')
            # 获取顶层模块名
            top_module = module_name.split('.')[0] if '.' in module_name else module_name
            
            if top_module not in module_stats:
                module_stats[top_module] = {'params': 0, 'tensors': 0, 'memory_mb': 0}
            
            tensor_info = analyze_tensor(value)
            module_stats[top_module]['params'] += tensor_info['numel']
            module_stats[top_module]['tensors'] += 1
            module_stats[top_module]['memory_mb'] += tensor_info['memory_mb']
            total_params += tensor_info['numel']
    
    return total_params, module_stats

# Load the model file
model_path = 'GLORY_MINDsmall_default_auc0.6775446534156799.pth'
print(f"Loading model from: {model_path}")
print(f"File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

try:
    model = torch.load(model_path, map_location='cpu')
    print("\nTop-level keys in the model file:")
    for key in model.keys():
        print(f"  - {key}")
    
    print("\n" + "="*60)
    print("MODEL COMPONENTS ANALYSIS")
    print("="*60)
    
    for key, value in model.items():
        print(f"\n[{key}]")
        if hasattr(value, 'shape'):
            tensor_info = analyze_tensor(value)
            print(f"  Type: Tensor")
            print(f"  Shape: {tensor_info['shape']}")
            print(f"  Parameters: {tensor_info['numel']:,}")
            print(f"  Memory: {tensor_info['memory_mb']:.2f} MB")
            
        elif isinstance(value, dict):
            print(f"  Type: Dictionary with {len(value)} items")
            
            # 如果是state_dict，进行模块级分析
            if 'state_dict' in key.lower() or any('weight' in k or 'bias' in k for k in value.keys()):
                print("  → Model State Dict Detected")
                total_params, module_stats = analyze_state_dict_summary(value)
                
                print(f"\n  OVERALL STATISTICS:")
                print(f"     Total Parameters: {total_params:,}")
                total_memory = sum(stats['memory_mb'] for stats in module_stats.values())
                print(f"     Total Memory: {total_memory:.2f} MB")
                
                print(f"\n  MODULE BREAKDOWN:")
                print(f"     {'Module':<20} {'Parameters':<12} {'Tensors':<8} {'Memory(MB)':<10}")
                print(f"     {'-'*20} {'-'*12} {'-'*8} {'-'*10}")
                
                # 按参数数量排序
                sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['params'], reverse=True)
                for module_name, stats in sorted_modules:
                    print(f"     {module_name:<20} {stats['params']:<12,} {stats['tensors']:<8} {stats['memory_mb']:<10.2f}")
            else:
                # 显示字典的前几个键
                sample_keys = list(value.keys())[:5]
                print(f"  Sample keys: {sample_keys}")
                if len(value) > 5:
                    print(f"  ... and {len(value) - 5} more keys")
        
        elif isinstance(value, (list, tuple)):
            print(f"  Type: {type(value).__name__} with {len(value)} items")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
        
        else:
            print(f"  Type: {type(value).__name__}")
            value_str = str(value)
            if len(value_str) > 100:
                print(f"  Value: {value_str[:100]}...")
            else:
                print(f"  Value: {value_str}")
            
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()