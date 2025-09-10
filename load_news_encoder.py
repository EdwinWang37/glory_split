import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.component.news_encoder import NewsEncoder
from models.base.layers import *

class NewsVectorizer:
    def __init__(self, model_path, cfg=None):
        """
        初始化新闻向量化器
        
        Args:
            model_path: 模型文件路径
            cfg: 配置对象（如果没有提供，将使用默认配置）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 如果没有提供配置，创建一个默认配置
        if cfg is None:
            cfg = self._create_default_config()
        
        self.cfg = cfg
        
        # 初始化NewsEncoder
        # 注意：这里需要根据实际的glove_emb来初始化
        # 从checkpoint中提取词汇表大小
        vocab_size = self._extract_vocab_size(checkpoint)
        
        self.news_encoder = NewsEncoder(cfg, glove_emb=vocab_size)
        
        # 加载local_news_encoder的权重
        self._load_news_encoder_weights(checkpoint)
        
        self.news_encoder.to(self.device)
        self.news_encoder.eval()
        
        print(f"NewsEncoder loaded successfully on {self.device}")
    
    def _create_default_config(self):
        """创建默认配置"""
        class Config:
            def __init__(self):
                self.model = type('obj', (object,), {
                    'word_emb_dim': 300,
                    'head_num': 8,   # 从新checkpoint分析结果得出：400 / 50 = 8
                    'head_dim': 50,  # 每个头的维度
                    'title_size': 30,
                    'abstract_size': 50,
                    'attention_hidden_dim': 200,  # 注意力池化层的隐藏维度
                    'attention_layer_num': 1  # 新checkpoint只有一个attention层
                })()
                self.dataset = type('obj', (object,), {
                    'dataset_lang': 'chinese'  # 根据实际情况调整
                })()
                self.dropout_probability = 0.2
        
        return Config()
    
    def _extract_vocab_size(self, checkpoint):
        """从checkpoint中提取词汇表大小"""
        # 查找local_news_encoder的word_encoder权重
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                for sub_key in checkpoint[key].keys():
                    if 'local_news_encoder.word_encoder.weight' in sub_key:
                        vocab_size = checkpoint[key][sub_key].shape[0]
                        print(f"Detected vocabulary size: {vocab_size}")
                        return vocab_size - 1  # 减1因为padding_idx=0
        
        # 如果没有找到，使用默认值
        print("Could not detect vocabulary size, using default: 50000")
        return 50000
    
    def _load_news_encoder_weights(self, checkpoint):
        """加载NewsEncoder的权重"""
        # 提取local_news_encoder的权重
        news_encoder_state_dict = {}
        
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                for param_key, param_value in checkpoint[key].items():
                    if param_key.startswith('local_news_encoder.'):
                        # 移除'local_news_encoder.'前缀
                        new_key = param_key.replace('local_news_encoder.', '')
                        news_encoder_state_dict[new_key] = param_value
        
        if news_encoder_state_dict:
            print(f"Found {len(news_encoder_state_dict)} parameters for NewsEncoder")
            # 加载权重
            missing_keys, unexpected_keys = self.news_encoder.load_state_dict(news_encoder_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        else:
            print("Warning: No NewsEncoder weights found in checkpoint")
    
    def vectorize_news(self, news_text_ids, mask=None):
        """
        对新闻进行向量化
        
        Args:
            news_text_ids: 新闻文本的token IDs，形状为 [batch_size, news_num, sequence_length]
                          或 [news_num, sequence_length] 或 [sequence_length]
            mask: 可选的mask，形状应与news_text_ids的前两个维度匹配
        
        Returns:
            新闻向量，形状为 [batch_size, news_num, news_dim]
        """
        with torch.no_grad():
            # 确保输入是tensor
            if not isinstance(news_text_ids, torch.Tensor):
                news_text_ids = torch.tensor(news_text_ids)
            
            # 移动到设备
            news_text_ids = news_text_ids.to(self.device)
            
            # 处理输入维度
            original_shape = news_text_ids.shape
            
            if len(original_shape) == 1:
                # [sequence_length] -> [1, 1, sequence_length]
                news_text_ids = news_text_ids.unsqueeze(0).unsqueeze(0)
            elif len(original_shape) == 2:
                # [news_num, sequence_length] -> [1, news_num, sequence_length]
                news_text_ids = news_text_ids.unsqueeze(0)
            
            # 确保序列长度匹配title_size
            seq_len = news_text_ids.shape[-1]
            if seq_len < self.cfg.model.title_size:
                # 填充到title_size
                padding = torch.zeros(news_text_ids.shape[:-1] + (self.cfg.model.title_size - seq_len,), 
                                    dtype=news_text_ids.dtype, device=self.device)
                news_text_ids = torch.cat([news_text_ids, padding], dim=-1)
            elif seq_len > self.cfg.model.title_size:
                # 截断到title_size
                news_text_ids = news_text_ids[..., :self.cfg.model.title_size]
            
            # 创建完整的新闻输入（包括其他字段的占位符）
            batch_size, news_num, title_size = news_text_ids.shape
            
            # 创建其他字段的占位符（abstract, category, subcategory, entity）
            abstract_placeholder = torch.zeros(batch_size, news_num, 5, dtype=torch.long, device=self.device)
            category_placeholder = torch.zeros(batch_size, news_num, 1, dtype=torch.long, device=self.device)
            subcategory_placeholder = torch.zeros(batch_size, news_num, 1, dtype=torch.long, device=self.device)
            entity_placeholder = torch.zeros(batch_size, news_num, 1, dtype=torch.long, device=self.device)
            
            # 拼接所有字段
            news_input = torch.cat([
                news_text_ids,  # title
                abstract_placeholder,  # abstract (5)
                category_placeholder,  # category (1)
                subcategory_placeholder,  # subcategory (1)
                entity_placeholder  # entity (1)
            ], dim=-1)
            
            # 进行向量化
            news_vectors = self.news_encoder(news_input, mask)
            
            # 根据原始输入形状调整输出
            if len(original_shape) == 1:
                return news_vectors.squeeze(0).squeeze(0)  # [news_dim]
            elif len(original_shape) == 2:
                return news_vectors.squeeze(0)  # [news_num, news_dim]
            else:
                return news_vectors  # [batch_size, news_num, news_dim]


def load_real_news_data(data_dir="./", load_all=False):
    """
    加载真实的新闻数据
    
    Args:
        data_dir: 数据目录路径
        load_all: 是否加载所有新闻数据
    
    Returns:
        news_data: 包含新闻token IDs的列表
        news_titles: 原始新闻标题列表（用于显示）
    """
    import pickle
    from pathlib import Path
    
    try:
        # 尝试加载预处理的新闻数据
        train_dir = Path(data_dir) / "train" if (Path(data_dir) / "train").exists() else Path(data_dir)
        
        # 加载新闻token数据
        news_token_file = train_dir / "nltk_token_news.bin"
        if news_token_file.exists():
            print(f"Loading real news data from: {news_token_file}")
            news_input = pickle.load(open(news_token_file, "rb"))
            
            # 加载新闻字典（用于获取原始标题）
            news_dict_file = train_dir / "news_dict.bin"
            news_raw_file = train_dir / "nltk_news.bin"
            
            news_titles = []
            if news_raw_file.exists():
                news_raw = pickle.load(open(news_raw_file, "rb"))
                # 按照新闻字典的顺序获取标题
                if news_dict_file.exists():
                    news_dict = pickle.load(open(news_dict_file, "rb"))
                    for news_id in news_dict:
                        if news_id in news_raw and not news_id.startswith('FAKE_'):
                            title_tokens = news_raw[news_id][0]  # 标题token列表
                            news_titles.append(' '.join(title_tokens))
                        else:
                            news_titles.append("N/A")
            
            # 如果加载所有数据，直接返回完整的新闻数据
            if load_all:
                news_data = [news_input[i].tolist() for i in range(len(news_input))]
                print(f"Successfully loaded all {len(news_data)} news articles")
                return news_data, news_titles
            else:
                 # 获取真实新闻的token IDs（跳过虚假新闻）
                 news_data = []
                 count = 0
                 max_news = 1000  # 默认最大数量
                 for i in range(min(len(news_input), max_news)):
                     if count >= max_news:
                         break
                     # 检查是否为有效的新闻数据（非全零）
                     if np.sum(news_input[i]) > 0:
                         news_data.append(news_input[i].tolist())
                         count += 1
                 
                 print(f"Successfully loaded {len(news_data)} real news articles")
                 return news_data, news_titles
            
        else:
            print(f"News token file not found: {news_token_file}")
            return None, None
            
    except Exception as e:
        print(f"Error loading real news data: {e}")
        return None, None


def load_user_behavior_data(data_dir="./", user_index=0):
    """
    加载用户行为数据，获取指定用户点击过的新闻
    
    Args:
        data_dir: 数据目录路径
        user_index: 用户索引
    
    Returns:
        user_clicked_news: 用户点击过的新闻ID列表
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        # 查找行为数据文件
        train_dir = Path(data_dir) / "train" if (Path(data_dir) / "train").exists() else Path(data_dir)
        behavior_file = train_dir / "behaviors_np4_0.tsv"
        
        if not behavior_file.exists():
            print(f"行为数据文件不存在: {behavior_file}")
            return None
        
        print(f"加载用户行为数据: {behavior_file}")
        
        # 读取行为数据
        behaviors = pd.read_csv(behavior_file, sep='\t', header=None, 
                               names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        
        print(f"总共有 {len(behaviors)} 条行为记录")
        print(f"总共有 {behaviors['user_id'].nunique()} 个用户")
        
        # 获取指定用户的数据
        if user_index >= len(behaviors):
            print(f"用户索引 {user_index} 超出范围，使用第一个用户")
            user_index = 0
        
        user_data = behaviors.iloc[user_index]
        user_id = user_data['user_id']
        history = user_data['history']
        
        print(f"\n选择用户: {user_id} (索引: {user_index})")
        
        # 解析用户历史点击新闻
        if pd.isna(history) or history == '':
            print("该用户没有历史点击记录")
            return None
        
        clicked_news = history.split(' ')
        print(f"用户点击过 {len(clicked_news)} 条新闻")
        
        return clicked_news
        
    except Exception as e:
        print(f"加载用户行为数据时出错: {e}")
        return None


def get_news_vectors_for_user(data_dir="./", user_index=0):
    """
    为指定用户的所有点击新闻生成向量
    
    Args:
        data_dir: 数据目录路径
        user_index: 用户索引
    """
    # 模型路径
    model_path = 'GLORY_MINDsmall_default_auc0.6760649681091309.pth'
    
    # 初始化向量化器
    print("=== 初始化新闻向量化器 ===")
    vectorizer = NewsVectorizer(model_path)
    
    # 加载用户行为数据
    print("\n=== 加载用户行为数据 ===")
    clicked_news_ids = load_user_behavior_data(data_dir, user_index)
    
    if clicked_news_ids is None or len(clicked_news_ids) == 0:
        print("无法获取用户点击的新闻，程序退出")
        return
    
    # 加载新闻数据
    print("\n=== 加载新闻数据 ===")
    real_news_data, real_news_titles = load_real_news_data(load_all=True)  # 加载所有新闻
    
    if real_news_data is None:
        print("无法加载新闻数据，程序退出")
        return
    
    # 加载新闻字典以获取新闻ID到索引的映射
    import pickle
    from pathlib import Path
    
    train_dir = Path(data_dir) / "train" if (Path(data_dir) / "train").exists() else Path(data_dir)
    news_dict_file = train_dir / "news_dict.bin"
    
    news_id_to_index = {}
    if news_dict_file.exists():
        news_dict = pickle.load(open(news_dict_file, "rb"))
        news_id_to_index = {news_id: idx for idx, news_id in enumerate(news_dict)}
        print(f"加载了 {len(news_id_to_index)} 条新闻的ID映射")
    
    # 为用户点击的每条新闻生成向量
    print(f"\n=== 为用户点击的 {len(clicked_news_ids)} 条新闻生成向量 ===")
    
    user_news_vectors = []
    valid_news_count = 0
    
    for i, news_id in enumerate(clicked_news_ids):
        try:
            # 获取新闻在数据中的索引
            if news_id in news_id_to_index:
                news_index = news_id_to_index[news_id]
                if news_index < len(real_news_data):
                    # 获取新闻的token IDs
                    news_tokens = real_news_data[news_index]
                    
                    # 生成新闻向量
                    news_vector = vectorizer.vectorize_news(news_tokens)
                    user_news_vectors.append(news_vector.cpu().numpy())
                    
                    # 打印新闻信息和向量
                    title = real_news_titles[news_index] if real_news_titles and news_index < len(real_news_titles) else "N/A"
                    print(f"\n新闻 {i+1}/{len(clicked_news_ids)}:")
                    print(f"  新闻ID: {news_id}")
                    print(f"  标题: {title[:100]}..." if len(title) > 100 else f"  标题: {title}")
                    print(f"  向量形状: {news_vector.shape}")
                    print(f"  向量前10个元素: {news_vector[:10].tolist()}")
                    print(f"  向量后10个元素: {news_vector[-10:].tolist()}")
                    
                    valid_news_count += 1
                else:
                    print(f"新闻 {news_id} 的索引 {news_index} 超出数据范围")
            else:
                print(f"新闻 {news_id} 不在新闻字典中")
                
        except Exception as e:
            print(f"处理新闻 {news_id} 时出错: {e}")
    
    print(f"\n=== 总结 ===")
    print(f"用户总共点击了 {len(clicked_news_ids)} 条新闻")
    print(f"成功生成了 {valid_news_count} 条新闻的向量")
    
    if user_news_vectors:
        user_news_vectors = np.array(user_news_vectors)
        print(f"所有新闻向量的形状: {user_news_vectors.shape}")
        print(f"向量维度: {user_news_vectors.shape[1]}")
        print(f"向量统计信息:")
        print(f"  均值: {np.mean(user_news_vectors):.6f}")
        print(f"  标准差: {np.std(user_news_vectors):.6f}")
        print(f"  最小值: {np.min(user_news_vectors):.6f}")
        print(f"  最大值: {np.max(user_news_vectors):.6f}")


def example_usage():
    """使用示例"""
    # 为第一个用户生成所有新闻向量
    get_news_vectors_for_user(user_index=0)


if __name__ == "__main__":
    example_usage()