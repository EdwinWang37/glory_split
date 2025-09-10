import torch
import os

def test_new_checkpoint():
    """测试新的checkpoint文件"""
    file_path = 'GLORY_MINDsmall_default_auc0.6760649681091309.pth'
    
    print(f"测试文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # 尝试加载文件
        checkpoint = torch.load(file_path, map_location='cpu')
        print("✅ 文件加载成功")
        print(f"顶层键: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print(f"模型参数数量: {len(model_state)}")
            
            # 检查是否有local_news_encoder
            encoder_keys = [k for k in model_state.keys() if 'local_news_encoder' in k]
            print(f"local_news_encoder参数数量: {len(encoder_keys)}")
            
            if encoder_keys:
                print("前5个local_news_encoder参数:")
                for key in encoder_keys[:5]:
                    print(f"  {key}: {model_state[key].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件加载失败: {str(e)}")
        print("\n可能的原因:")
        print("1. 文件在传输过程中损坏")
        print("2. 文件保存时出现问题")
        print("3. 磁盘空间不足导致文件不完整")
        print("4. 文件格式不兼容")
        return False

if __name__ == "__main__":
    test_new_checkpoint()