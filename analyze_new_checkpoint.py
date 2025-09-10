import torch
import numpy as np

def analyze_checkpoint_structure():
    """分析新checkpoint文件的结构"""
    model_path = 'GLORY_MINDsmall_default_auc0.6760649681091309.pth'
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("\n=== Checkpoint结构分析 ===")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # 检查model_state_dict
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        print(f"\nModel state dict包含 {len(model_state)} 个参数")
        
        # 收集所有local_news_encoder相关的参数
        encoder_params = {}
        for param_key, param_value in model_state.items():
            if param_key.startswith('local_news_encoder.'):
                clean_key = param_key.replace('local_news_encoder.', '')
                encoder_params[clean_key] = param_value.shape
        
        print(f"\n=== local_news_encoder参数分析 ({len(encoder_params)}个参数) ===")
        
        # 按模块分组分析
        print("\n1. Word Encoder:")
        for key, shape in encoder_params.items():
            if 'word_encoder' in key:
                print(f"  {key}: {shape}")
        
        print("\n2. Attention层:")
        for key, shape in encoder_params.items():
            if key.startswith('attention.'):
                print(f"  {key}: {shape}")
        
        print("\n3. 其他参数:")
        for key, shape in encoder_params.items():
            if not any(x in key for x in ['word_encoder', 'attention.']):
                print(f"  {key}: {shape}")
        
        # 分析attention层的结构
        print("\n=== 推断模型结构 ===")
        
        # 从word_encoder推断词嵌入维度
        word_emb_shape = encoder_params.get('word_encoder.weight')
        if word_emb_shape:
            vocab_size, word_emb_dim = word_emb_shape
            print(f"词汇表大小: {vocab_size}")
            print(f"词嵌入维度: {word_emb_dim}")
        
        # 分析attention层
        att_q_shape = encoder_params.get('attention.module_1.W_Q.weight')
        if att_q_shape:
            out_dim, in_dim = att_q_shape
            print(f"\nAttention层:")
            print(f"  输入维度: {in_dim}")
            print(f"  输出维度: {out_dim}")
            
            # 推断head配置
            possible_heads = []
            for head_num in [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 100, 200]:
                if out_dim % head_num == 0:
                    head_dim = out_dim // head_num
                    possible_heads.append((head_num, head_dim))
            
            print(f"  可能的head配置: {possible_heads[:5]}")
        
        # 检查是否有其他attention相关的层
        print("\n=== 检查是否有其他attention层 ===")
        attention_layers = []
        for key in encoder_params.keys():
            if 'attention' in key or 'attetio' in key:
                attention_layers.append(key)
        
        print(f"所有attention相关的层: {attention_layers}")
        
        # 检查最终输出维度
        print("\n=== 推断最终输出维度 ===")
        # 查找可能的最后一层
        last_layers = []
        for key, shape in encoder_params.items():
            if any(x in key for x in ['last', 'final', 'output', 'fc']):
                last_layers.append((key, shape))
        
        if last_layers:
            print("可能的最后一层:")
            for key, shape in last_layers:
                print(f"  {key}: {shape}")
        else:
            print("未找到明显的最后一层，attention层可能直接输出最终结果")
            if att_q_shape:
                print(f"Attention层输出维度: {att_q_shape[0]}")
    
    else:
        print("未找到model_state_dict")

if __name__ == "__main__":
    analyze_checkpoint_structure()