#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成的虚假新闻生成器
结合VAE和新闻编码器，为推荐系统生成对抗性虚假新闻

主要功能：
1. 加载真实新闻编码器（NewsVectorizer）
2. 训练VAE生成器
3. 生成个性化虚假新闻向量
4. 支持多种生成策略（uniform_space_filling, anti_interest）
5. 提供分布均匀性验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import os

# 导入VAE生成器
from .vae_generator import VAE, FakeNewsGenerator

class IntegratedFakeNewsGenerator:
    """
    集成的虚假新闻生成器
    结合新闻编码器和VAE，生成高质量的对抗性虚假新闻
    
    Attributes:
        cfg: 配置对象
        device: 计算设备 ('cuda' 或 'cpu')
        news_encoder: 新闻编码器（NewsVectorizer或MockNewsEncoder）
        vae_generator: VAE生成器
        is_initialized: 是否已初始化
        news_vectors: 新闻ID到向量的映射
        real_news_vectors: 真实新闻向量
        real_news_ids: 真实新闻ID列表
    """
    
    def __init__(self, cfg, device='cuda'):
        """
        初始化生成器
        
        Args:
            cfg: 配置对象
            device: 计算设备，默认'cuda'
        """
        self.cfg = cfg
        self.device = device
        self.news_encoder = None
        self.vae_generator = None
        self.is_initialized = False
        
        # 初始化后设置的属性
        self.news_vectors = {}
        self.real_news_vectors = None
        self.real_news_ids = []
        self.news_input = None # Initialize news_input
        self.news_index = None # Initialize news_index
        
    def load_news_encoder(self, model_path, input_dim=None, output_dim=400):
        """加载真实的新闻编码器（从checkpoint），失败时回退到Mock编码器"""
        try:
            # 确保可以导入项目根目录下的向量化器
            project_root = Path(__file__).resolve().parents[2]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # 尝试导入NewsVectorizer
            from load_news_encoder import NewsVectorizer

            # 使用真实的NewsEncoder参数初始化向量化器
            print("[DEBUG] Before NewsVectorizer instantiation.")
            self.news_encoder = NewsVectorizer(model_path, cfg=self.cfg)
            print("[DEBUG] After NewsVectorizer instantiation.")
            print(f"Real NewsEncoder loaded from checkpoint: {model_path}")
            return True
        except Exception as e:
            import traceback
            print(f"Failed to load real NewsEncoder from checkpoint: {e}")
            traceback.print_exc()  # 打印详细错误堆栈
            print("Using Mock encoder as fallback.")
            # 回退：使用简单的线性编码器，保持接口一致
            class MockNewsEncoder:
                def __init__(self, device, input_dim, output_dim):
                    self.device = device
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim or 38, 200),
                        nn.ReLU(),
                        nn.Linear(200, output_dim)
                    ).to(device)

                def vectorize_news(self, news_tokens):
                    with torch.no_grad():
                        return self.encoder(news_tokens.float())

            self.news_encoder = MockNewsEncoder(self.device, input_dim, output_dim)
            return True
    
    def load_real_news_data(self, data_dir, mode='train'):
        """加载真实新闻数据并返回对应ID顺序（兼容两种路径结构）
        Returns:
            real_news_data: list of token tensors for real news
            real_news_indices: list of indices in original bin
            news_dict: mapping of news_id -> index
            real_news_ids: list of real news ids aligned with real_news_data order
        """
        try:
            # 兼容 data_dir/train/* 与 data_dir/* 两种结构
            news_candidates = [Path(data_dir) / f"{mode}/nltk_token_news.bin",
                               Path(data_dir) / "nltk_token_news.bin"]
            dict_candidates = [Path(data_dir) / f"{mode}/news_dict.bin",
                               Path(data_dir) / "news_dict.bin"]
            
            news_file, dict_file = None, None
            for nf in news_candidates:
                if nf.exists():
                    news_file = nf
                    break
            for df in dict_candidates:
                if df.exists():
                    dict_file = df
                    break
                    
            if news_file is None or dict_file is None:
                raise FileNotFoundError(f"Cannot find required files in {data_dir} (mode={mode})")
            
            print(f"Loading news data from: {news_file}")
            print(f"Loading news dict from: {dict_file}")
            
            # 加载新闻数据
            with open(news_file, 'rb') as f:
                news_input = pickle.load(f)
            
            # 加载新闻字典
            with open(dict_file, 'rb') as f:
                news_dict = pickle.load(f)
            
            self.news_input = news_input
            self.news_index = news_dict
            
            # 过滤出真实新闻（排除FAKE_开头的）
            real_news_data = []
            real_news_indices = []
            real_news_ids = []
            
            for news_id, news_index in news_dict.items():
                if not news_id.startswith('FAKE_') and news_index < len(news_input):
                    real_news_data.append(news_input[news_index])
                    real_news_indices.append(news_index)
                    real_news_ids.append(news_id)
            
            print(f"Loaded {len(real_news_data)} real news articles")
            return real_news_data, real_news_indices, news_dict, real_news_ids
            
        except FileNotFoundError as e:
            print(f"File not found error: {e}")
            return [], [], {}, []
        except Exception as e:
            print(f"Failed to load real news data: {e}")
            return [], [], {}, []
    
    def generate_real_news_vectors(self, real_news_data, batch_size=32):
        """生成真实新闻的向量表示（兼容真实NewsVectorizer与Mock）"""
        if self.news_encoder is None:
            raise ValueError("News encoder not loaded")

        all_vectors = []

        print("Generating real news vectors...")
        for i in tqdm(range(0, len(real_news_data), batch_size)):
            batch_data = real_news_data[i:i+batch_size]

            # 转换为tensor
            batch_tensor = torch.tensor(batch_data, dtype=torch.long).to(self.device)

            # 生成向量：NewsVectorizer 返回 [batch, dim] 或更高维
            with torch.no_grad():
                try:
                    batch_vectors = self.news_encoder.vectorize_news(batch_tensor)
                    if torch.isnan(batch_vectors).any() or torch.isinf(batch_vectors).any():
                        print(f"WARNING: MockNewsEncoder output contains NaN or Inf at batch {i}")
                except Exception as e:
                    print(f"Error in vectorize_news: {e}")
                    # 若 self.news_encoder 是向量器类（有 .news_encoder），走其内部接口
                    encoder = getattr(self.news_encoder, 'news_encoder', None)
                    if encoder is None:
                        raise
                    batch_vectors = self.news_encoder.vectorize_news(batch_tensor)

                # 统一为二维 [batch, dim]
                if isinstance(batch_vectors, torch.Tensor):
                    if batch_vectors.dim() == 1:
                        all_vectors.append(batch_vectors.unsqueeze(0).cpu())
                    elif batch_vectors.dim() == 2:
                        all_vectors.append(batch_vectors.cpu())
                    else:
                        all_vectors.append(batch_vectors.view(batch_vectors.shape[0], -1).cpu())
                else:
                    # 非tensor（不应发生），跳过
                    print(f"Warning: Unexpected output type from news encoder: {type(batch_vectors)}")
                    continue

        if not all_vectors:
            raise ValueError("No vectors generated from news data")

        # 合并所有向量
        real_vectors = torch.cat(all_vectors, dim=0)
        print(f"Generated {real_vectors.shape[0]} news vectors with dimension {real_vectors.shape[1]}")

        return real_vectors
    
    def get_news_vectors_by_ids(self, news_ids):
        """
        根据新闻ID列表获取对应的新闻向量。
        Args:
            news_ids: 新闻ID列表。
        Returns:
            一个包含新闻向量的列表。
        """
        vectors = []
        for news_id in news_ids:
            if news_id in self.news_vectors:
                vectors.append(self.news_vectors[news_id])
            else:
                # 如果找不到，可以返回一个默认向量或者抛出错误，这里选择返回零向量
                print(f"Warning: News ID {news_id} not found in news_vectors. Returning zero vector.")
                # 假设向量维度为 word_emb_dim，需要从配置中获取
                vectors.append(torch.zeros(self.cfg.model.word_emb_dim))
        return torch.stack(vectors).to(self.device)

    def initialize_vae(self, model_path=None, data_dir="./"):
        """初始化VAE生成器。"""
        print("Initializing VAE for fake news generation...")
        
        # 1. 加载真实新闻数据
        real_news_data, real_news_indices, news_dict, real_news_ids = self.load_real_news_data(data_dir, mode='train')
        
        if len(real_news_data) == 0:
            print("Error: No real news data loaded")
            return False
        
        # 2. 加载新闻编码器（优先使用真实checkpoint）
        if model_path is None:
            # 默认使用项目checkpoint（可根据cfg.path设置）
            project_root = Path(__file__).resolve().parents[2]
            model_path = str(project_root / 'checkpoint' / 'GLORY_MINDsmall_default_auc0.6760649681091309.pth')
        
        # 推断输入维度（仅作为Mock回退时使用）
        sample = real_news_data[0]
        try:
            token_dim = len(sample)
        except Exception:
            token_dim = np.array(sample).shape[-1]
        
        if not self.load_news_encoder(model_path, input_dim=token_dim, output_dim=400):
            print("Error: Failed to load news encoder")
            return False
        
        # 3. 生成真实新闻向量
        try:
            real_vectors = self.generate_real_news_vectors(real_news_data)
        except Exception as e:
            print(f"Error generating real news vectors: {e}")
            return False
        
        # 4. 初始化VAE生成器
        input_dim = real_vectors.shape[1]
        latent_dim = getattr(self.cfg, 'vae_latent_dim', 50)
        
        self.vae_generator = FakeNewsGenerator(
            input_dim=input_dim,
            latent_dim=latent_dim,
            device=self.device
        )
        
        # 5. 训练VAE
        print("Training VAE on real news vectors...")
        epochs = getattr(self.cfg, 'vae_training_epochs', 1000)
        self.vae_generator.train_vae(real_vectors.to(self.device), epochs=epochs)
        
        # 保存新闻向量映射，便于个性化生成时查找用户历史对应向量
        # real_news_ids 与 real_vectors 行顺序一致
        self.news_vectors = {nid: vec.cpu() for nid, vec in zip(real_news_ids, real_vectors)}
        self.real_news_vectors = real_vectors.cpu()
        self.real_news_ids = real_news_ids
        
        self.is_initialized = True
        print("VAE initialization completed successfully!")
        return True
    
    def generate_fake_news_for_user(self, user_id, user_history_vectors=None, num_fake_news=None):
        """
        为特定用户生成虚假新闻向量
        
        Args:
            user_id: 用户ID
            user_history_vectors: 用户历史点击新闻的向量（可选）
            num_fake_news: 要生成的虚假新闻数量
        
        Returns:
            fake_news_vectors: 生成的虚假新闻向量
        """
        if not self.is_initialized:
            raise ValueError("VAE not initialized. Please call initialize_vae() first.")
        
        if num_fake_news is None:
            num_fake_news = getattr(self.cfg, 'fake_news_per_user', 5)
        
        try:
            # 如果没有提供用户历史向量，生成随机用户兴趣
            if user_history_vectors is None:
                # 生成随机用户兴趣向量
                user_interest = torch.randn(1, self.vae_generator.input_dim).to(self.device)
            else:
                # 使用用户历史向量的平均值作为用户兴趣
                user_interest = torch.mean(user_history_vectors, dim=0, keepdim=True).to(self.device)
            
            # 生成虚假新闻
            iterations = getattr(self.cfg, 'vae_generation_iterations', 500)
            fake_news_vectors = self.vae_generator.generate_fake_news(
                user_interest, 
                num_samples=num_fake_news,
                num_iterations=iterations
            )
            
            return fake_news_vectors
            
        except Exception as e:
            print(f"Error generating fake news for user {user_id}: {e}")
            # 返回随机向量作为后备
            return torch.randn(num_fake_news, self.vae_generator.input_dim)
    
    def generate_fake_news_features(self, fake_news_ids, user_histories=None):
        """
        为给定的虚假新闻ID生成特征
        这个方法用于与现有的数据预处理流程集成
        
        Args:
            fake_news_ids: 虚假新闻ID列表
            user_histories: 用户历史数据（可选）
        
        Returns:
            fake_news_features: 虚假新闻特征字典
        """
        if not self.is_initialized:
            print("Warning: VAE not initialized, using random features")
            # 返回随机特征作为后备
            fake_news_features = {}
            for fake_id in fake_news_ids:
                fake_news_features[fake_id] = {
                    'title': torch.randint(0, 1000, (30,)),  # 随机标题token
                    'entity': torch.randint(0, 100, (5,)),   # 随机实体token
                    'vector': torch.randn(400)               # 随机向量
                }
            return fake_news_features
        
        # 使用VAE生成虚假新闻特征
        fake_news_features = {}
        
        # 为每个虚假新闻ID生成特征
        for i, fake_id in enumerate(fake_news_ids):
            # 从ID中提取用户信息（如果有的话）
            user_history = None
            if user_histories and fake_id.startswith('FAKE_'):
                # 尝试从ID中提取用户hash
                parts = fake_id.split('_')
                if len(parts) >= 3:
                    user_hash = parts[1]
                    if user_hash in user_histories:
                        user_history = user_histories[user_hash]
            
            # 生成虚假新闻向量
            if user_history and hasattr(self, 'news_vectors'):
                # 如果有用户历史，生成个性化的虚假新闻
                user_history_vectors = []
                for news_id in user_history:
                    if news_id in self.news_vectors:
                        user_history_vectors.append(self.news_vectors[news_id])
                
                if user_history_vectors:
                    user_history_tensor = torch.stack(user_history_vectors)
                    fake_vector = self.generate_fake_news_for_user(
                        user_id=f"user_{i}", 
                        user_history_vectors=user_history_tensor,
                        num_fake_news=1
                    )[0]
                else:
                    fake_vector = self.generate_fake_news_for_user(
                        user_id=f"user_{i}",
                        num_fake_news=1
                    )[0]
            else:
                fake_vector = self.generate_fake_news_for_user(
                    user_id=f"user_{i}",
                    num_fake_news=1
                )[0]
            
            # 生成对应的标题和实体token（这里使用简化的方法）
            fake_news_features[fake_id] = {
                'title': torch.randint(0, 1000, (30,)),  # 模拟标题token
                'entity': torch.randint(0, 100, (5,)),   # 模拟实体token
                'vector': fake_vector.cpu()              # VAE生成的向量
            }
        
        return fake_news_features
    
    def generate_personalized_fake_news(self, user_histories, num_fake_per_user=5, 
                                      generation_strategy='uniform_space_filling'):
        """为每个用户生成个性化的假新闻向量"""
        if not self.is_initialized:
            raise ValueError("VAE生成器未初始化或未训练")
        
        fake_news_dict = {}
        
        for user_hash, news_ids in user_histories.items():
            if not news_ids:
                continue
                
            # 获取用户的真实新闻向量
            user_vectors = []
            for news_id in news_ids:
                if hasattr(self, 'news_vectors') and news_id in self.news_vectors:
                    user_vectors.append(self.news_vectors[news_id])
            
            if not user_vectors:
                continue
                
            user_vectors_tensor = torch.stack(user_vectors)
            
            # 根据生成策略选择不同的生成方法
            if generation_strategy == 'uniform_space_filling':
                # 使用新的空间填充算法
                if hasattr(self.vae_generator, 'generate_uniform_space_filling_vectors'):
                    fake_vectors = self.vae_generator.generate_uniform_space_filling_vectors(
                        user_vectors_tensor, 
                        num_fake_per_user=num_fake_per_user,
                        diversity_weight=1.0,
                        coverage_weight=1.5,
                        num_iterations=800
                    )
                else:
                    # 回退到基本生成方法
                    user_interest = user_vectors_tensor.mean(dim=0)
                    fake_vectors = self.vae_generator.generate_fake_news(
                        user_interest.unsqueeze(0), 
                        num_samples=num_fake_per_user
                    )
            elif generation_strategy == 'anti_interest':
                # 使用原来的反兴趣生成方法
                user_interest = user_vectors_tensor.mean(dim=0)
                fake_vectors = self.vae_generator.generate_fake_news(
                    user_interest.unsqueeze(0), 
                    num_samples=num_fake_per_user
                )
            else:
                raise ValueError(f"未知的生成策略: {generation_strategy}")
            
            # 为每个假新闻分配ID
            for i, fake_vector in enumerate(fake_vectors):
                fake_news_id = f"FAKE_{user_hash}_{i}"
                fake_news_dict[fake_news_id] = fake_vector
        
        return fake_news_dict
    
    def generate_global_uniform_fake_news(self, user_histories, total_fake_news=1000, 
                                        pca_components=50):
        """
        生成全局均匀分布的假新闻向量，确保在PCA降维后均匀覆盖整个向量空间
        
        Args:
            user_histories: 用户历史数据
            total_fake_news: 总共生成的假新闻数量
            pca_components: PCA降维维度
        
        Returns:
            fake_news_dict: 假新闻字典
        """
        if not self.is_initialized:
            raise ValueError("VAE生成器未初始化或未训练")
        
        # 1. 收集所有真实新闻向量
        all_real_vectors = []
        if hasattr(self, 'news_vectors'):
            for news_id, vector in self.news_vectors.items():
                if not news_id.startswith('FAKE_'):
                    all_real_vectors.append(vector)
        
        if not all_real_vectors:
            raise ValueError("没有找到真实新闻向量")
        
        all_real_tensor = torch.stack(all_real_vectors)
        
        # 2. 使用PCA均匀生成方法
        if hasattr(self.vae_generator, 'generate_pca_uniform_vectors'):
            fake_vectors = self.vae_generator.generate_pca_uniform_vectors(
                all_real_tensor, 
                num_fake_total=total_fake_news,
                pca_components=pca_components
            )
        else:
            # 回退到基本生成方法
            fake_vectors = self.vae_generator.generate_fake_news(
                all_real_tensor.mean(dim=0).unsqueeze(0), 
                num_samples=total_fake_news
            )
        
        # 3. 分配假新闻ID
        fake_news_dict = {}
        for i, fake_vector in enumerate(fake_vectors):
            fake_news_id = f"FAKE_GLOBAL_{i}"
            fake_news_dict[fake_news_id] = fake_vector
        
        return fake_news_dict
    
    def validate_distribution_uniformity(self, real_vectors, fake_vectors, pca_components=50):
        """
        验证真实和虚假向量在PCA降维后的分布均匀性
        
        Args:
            real_vectors: 真实向量
            fake_vectors: 虚假向量
            pca_components: PCA降维维度
        
        Returns:
            uniformity_metrics: 均匀性指标
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.neighbors import NearestNeighbors
            import numpy as np
            
            # 合并所有向量
            all_vectors = torch.cat([real_vectors, fake_vectors], dim=0)
            all_np = all_vectors.cpu().numpy()
            
            # PCA降维
            pca = PCA(n_components=pca_components)
            all_pca = pca.fit_transform(all_np)
            
            real_pca = all_pca[:len(real_vectors)]
            fake_pca = all_pca[len(real_vectors):]
            
            # 计算均匀性指标
            metrics = {}
            
            # 1. 最近邻距离分布
            nbrs = NearestNeighbors(n_neighbors=2).fit(all_pca)
            distances, _ = nbrs.kneighbors(all_pca)
            nn_distances = distances[:, 1]  # 排除自己
            
            metrics['mean_nn_distance'] = np.mean(nn_distances)
            metrics['std_nn_distance'] = np.std(nn_distances)
            metrics['uniformity_coefficient'] = metrics['std_nn_distance'] / metrics['mean_nn_distance']
            
            # 2. 空间覆盖率
            # 计算凸包体积比例
            from scipy.spatial import ConvexHull
            try:
                real_hull = ConvexHull(real_pca)
                all_hull = ConvexHull(all_pca)
                metrics['coverage_ratio'] = real_hull.volume / all_hull.volume
            except:
                metrics['coverage_ratio'] = 0.0
            
            # 3. 分布相似性 (Wasserstein距离)
            from scipy.stats import wasserstein_distance
            
            # 在每个PCA维度上计算分布相似性
            wasserstein_distances = []
            for dim in range(min(5, pca_components)):  # 只检查前5个主成分
                wd = wasserstein_distance(real_pca[:, dim], fake_pca[:, dim])
                wasserstein_distances.append(wd)
            
            metrics['mean_wasserstein_distance'] = np.mean(wasserstein_distances)
            
            return metrics
        except ImportError as e:
            print(f"Warning: Required packages not available for uniformity validation: {e}")
            return {'error': 'Required packages not available'}
    
    def generate_personalized_fake_news_vectors(self, num_fake_news, user_id=None, user_history_vectors=None):
        """
        生成个性化虚假新闻向量并返回其ID。
        如果未提供user_id或user_history_vectors，则生成非个性化的假新闻。
        """
        if not self.is_initialized:
            raise ValueError("VAE生成器未初始化或未训练")

        generated_fake_news_ids = []
        generated_fake_news_vectors = []
        
        # 如果没有提供用户历史，则生成随机用户兴趣
        if user_history_vectors is None:
            # 这里的user_id可以是一个占位符，因为我们没有实际的用户ID
            # 我们可以调用 generate_fake_news_for_user 来生成向量
            fake_vectors = self.generate_fake_news_for_user(user_id="dummy_user", num_fake_news=num_fake_news)
        else:
            # 如果提供了用户历史向量，则使用它们来生成个性化假新闻
            fake_vectors = self.generate_fake_news_for_user(user_id=user_id, user_history_vectors=user_history_vectors, num_fake_news=num_fake_news)

        for i, fake_vector in enumerate(fake_vectors):
            # 生成唯一的假新闻ID
            # 这里的ID需要确保在整个会话中是唯一的
            fake_news_id = f"FAKE_GENERATED_{os.getpid()}_{id(self)}_{len(self.news_vectors) + i}"
            self.news_vectors[fake_news_id] = fake_vector.cpu() # Temporarily store in news_vectors for consistency with other parts of the code
            generated_fake_news_ids.append(fake_news_id)
            generated_fake_news_vectors.append(fake_vector.cpu())
            
        return generated_fake_news_ids, generated_fake_news_vectors

    def generate_uniform_vectors(self, num_samples, method='adversarial'):
        # 这是一个兼容测试文件的占位符方法
        # 实际调用 generate_personalized_fake_news_vectors
        return self.generate_personalized_fake_news_vectors(num_fake_news=num_samples)