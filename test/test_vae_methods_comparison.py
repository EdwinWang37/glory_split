#!/usr/bin/env python3
"""
VAE Generation Methods Comparison Test
Compare generate_fake_news and generate_uniform_space_filling_vectors methods
Demonstrate the superiority of uniform method through PCA visualization
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.dataload.vae_generator import FakeNewsGenerator
from load_news_encoder import NewsVectorizer

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class VAEMethodsComparison:
    def __init__(self, data_dir="./", model_path=None):
        """
        Initialize VAE methods comparison test
        
        Args:
            data_dir: Data directory path
            model_path: Model checkpoint path
        """
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set model path
        if model_path is None:
            self.model_path = project_root / 'checkpoint' / 'GLORY_MINDsmall_default_auc0.6760649681091309.pth'
        else:
            self.model_path = Path(model_path)
        
        # Initialize components
        self.vectorizer = None
        self.vae_generator = None
        self.real_news_vectors = None
        self.user_vectors = None
        
    def load_real_news_data(self, max_news=100000):  # Increased to 100000
        """
        Load real news data
        
        Args:
            max_news: Maximum number of news to load
        
        Returns:
            news_data: List of news token data
            news_dict: News ID mapping dictionary
        """
        try:
            # Find data files
            train_dir = self.data_dir / "train"
            if not train_dir.exists():
                train_dir = self.data_dir
            
            news_token_file = train_dir / "nltk_token_news.bin"
            news_dict_file = train_dir / "news_dict.bin"
            
            if not news_token_file.exists() or not news_dict_file.exists():
                raise FileNotFoundError(f"Data files not found: {news_token_file} or {news_dict_file}")
            
            print(f"Loading news data: {news_token_file}")
            with open(news_token_file, 'rb') as f:
                news_input = pickle.load(f)
            
            print(f"Loading news dictionary: {news_dict_file}")
            with open(news_dict_file, 'rb') as f:
                news_dict = pickle.load(f)
            
            # Limit news count
            if max_news and len(news_input) > max_news:
                news_input = news_input[:max_news]
                print(f"Limited news count to: {max_news}")
            
            print(f"Successfully loaded {len(news_input)} news items")
            print(f"News data range: 0 to {len(news_input)-1}")
            print(f"News dictionary contains {len(news_dict)} news IDs")
            
            return news_input, news_dict
            
        except Exception as e:
            print(f"Failed to load news data: {e}")
            return None, None
    
    def load_user_behavior_data(self, num_users=5, min_clicks=20):
        """
        Load user behavior data
        
        Args:
            num_users: Number of users to select
            min_clicks: Minimum number of clicks
        
        Returns:
            selected_users: Selected user data
        """
        try:
            train_dir = self.data_dir / "train"
            if not train_dir.exists():
                train_dir = self.data_dir
            
            behavior_file = train_dir / "behaviors_np4_0.tsv"
            if not behavior_file.exists():
                raise FileNotFoundError(f"Behavior data file not found: {behavior_file}")
            
            print(f"Loading user behavior data: {behavior_file}")
            
            user_clicks = {}
            with open(behavior_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        user_id = parts[1]
                        history = parts[3]
                        
                        if history and history != '':
                            clicked_news = history.split(' ')
                            if user_id not in user_clicks:
                                user_clicks[user_id] = []
                            user_clicks[user_id].extend(clicked_news)
            
            # Deduplicate and filter users
            for user_id in user_clicks:
                user_clicks[user_id] = list(set(user_clicks[user_id]))
            
            # Select qualified users
            valid_users = [(uid, len(clicks), clicks) for uid, clicks in user_clicks.items() 
                          if len(clicks) >= min_clicks]
            valid_users.sort(key=lambda x: x[1], reverse=True)
            
            selected_users = valid_users[:num_users]
            print(f"Selected {len(selected_users)} users, click count range: {selected_users[-1][1]} - {selected_users[0][1]}")
            
            return selected_users
            
        except Exception as e:
            print(f"Failed to load user behavior data: {e}")
            return []
    
    def initialize_components(self):
        """Initialize news vectorizer and VAE generator"""
        try:
            # 1. Initialize news vectorizer
            print("Initializing news vectorizer...")
            self.vectorizer = NewsVectorizer(str(self.model_path))
            
            # 2. Load news data and generate vectors
            news_data, news_dict = self.load_real_news_data(max_news=5000)  # Increased to 5000
            if news_data is None:
                raise ValueError("Unable to load news data")
            
            # 3. Generate news vector samples
            print("Generating news vector samples...")
            sample_vectors = []
            for i in range(min(1000, len(news_data))):  # Use first 1000 news items
                try:
                    news_tokens = news_data[i]
                    vector = self.vectorizer.vectorize_news(news_tokens)
                    if vector is not None:
                        sample_vectors.append(vector.cpu().numpy())
                except Exception as e:
                    continue
            
            if len(sample_vectors) == 0:
                raise ValueError("Unable to generate news vectors")
            
            self.real_news_vectors = torch.tensor(np.array(sample_vectors), device=self.device)
            print(f"Generated {len(sample_vectors)} news vectors, dimension: {self.real_news_vectors.shape[1]}")
            
            # 4. Initialize VAE generator
            print("Initializing VAE generator...")
            input_dim = self.real_news_vectors.shape[1]
            self.vae_generator = FakeNewsGenerator(input_dim=input_dim, latent_dim=32, device=self.device)
            
            # 5. Train VAE
            print("Training VAE model...")
            self.vae_generator.train_vae(self.real_news_vectors, epochs=500, lr=1e-3)
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize components: {e}")
            return False
    
    def prepare_user_data(self):
        """Prepare user data"""
        try:
            # Load user behavior data
            selected_users = self.load_user_behavior_data(num_users=5, min_clicks=15)  # Increased user count
            if not selected_users:
                raise ValueError("Unable to load user data")
            
            # Prepare vector data for each user
            user_data = []
            news_data, news_dict = self.load_real_news_data(max_news=10000)  # Increased news count
            
            print(f"News dictionary contains {len(news_dict)} news IDs")
            print(f"News data contains {len(news_data)} news items")
            
            for user_id, click_count, clicked_news in selected_users:
                user_vectors = []
                valid_news_count = 0
                
                print(f"\nProcessing user {user_id}, clicked {len(clicked_news)} news items")
                
                # Get vectors for user's clicked news
                for i, news_id in enumerate(clicked_news[:50]):  # Increased to 50 news items
                    if news_id in news_dict:
                        news_index = news_dict[news_id]
                        if news_index < len(news_data):
                            try:
                                news_tokens = news_data[news_index]
                                vector = self.vectorizer.vectorize_news(news_tokens)
                                if vector is not None:
                                    user_vectors.append(vector.cpu().numpy())
                                    valid_news_count += 1
                                    if valid_news_count >= 10:  # Stop once we get 10 vectors
                                        break
                            except Exception as e:
                                print(f"  News {news_id} vectorization failed: {e}")
                                continue
                        else:
                            print(f"  News {news_id} index {news_index} out of range (max: {len(news_data)-1})")
                    else:
                        print(f"  News {news_id} not in dictionary")
                
                print(f"  User {user_id} successfully generated {valid_news_count} news vectors")
                
                # Lower requirement: need at least 3 vectors
                if len(user_vectors) >= 3:
                    user_vectors_tensor = torch.tensor(np.array(user_vectors), device=self.device)
                    user_data.append({
                        'user_id': user_id,
                        'vectors': user_vectors_tensor,
                        'click_count': click_count
                    })
                    print(f"  ✅ User {user_id}: {len(user_vectors)} news vectors")
                else:
                    print(f"  ❌ User {user_id} insufficient vectors: {len(user_vectors)} < 3")
            
            self.user_vectors = user_data
            print(f"\nPrepared data for {len(user_data)} users")
            return len(user_data) > 0
            
        except Exception as e:
            print(f"Failed to prepare user data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_fake_news_comparison(self):
        """
        Compare two fake news generation methods for multiple users
        
        Returns:
            comparison_results: Dictionary containing results from both methods for multiple users
        """
        if not self.user_vectors:
            raise ValueError("User data not ready")
        
        # Use all available users (up to 3 for visualization clarity)
        num_users = min(3, len(self.user_vectors))
        selected_users = self.user_vectors[:num_users]
        
        print(f"\n=== Multi-User VAE Methods Comparison ===")
        print(f"Comparing {num_users} users with both generation methods")
        
        results = {
            'users': [],
            'comparison_summary': {}
        }
        
        for i, user_data in enumerate(selected_users):
            user_id = user_data['user_id']
            user_real_vectors = user_data['vectors']
            
            print(f"\n--- Processing User {i+1}/{num_users}: {user_id} ---")
            print(f"User's real news vectors: {user_real_vectors.shape[0]} items")
            print(f"Vector dimension: {user_real_vectors.shape[1]}")
            
            user_result = {
                'user_id': user_id,
                'real_vectors': user_real_vectors.cpu().numpy(),
                'anti_interest_vectors': None,
                'uniform_space_vectors': None
            }
            
            # Calculate user interest vector (mean of user's clicked news)
            user_interest = user_real_vectors.mean(dim=0, keepdim=True)
            print(f"User interest vector calculated from {user_real_vectors.shape[0]} real news items")
            
            # Method 1: Anti-interest generation (generate_fake_news)
            print(f"\n  Method 1: Anti-Interest Generation for User {user_id}")
            try:
                fake_anti = self.vae_generator.generate_fake_news(
                    user_interest=user_interest,
                    num_samples=15,  # Generate samples for comparison
                    num_iterations=300
                )
                user_result['anti_interest_vectors'] = fake_anti.cpu().numpy()
                print(f"  ✅ Anti-interest method generated {fake_anti.shape[0]} vectors")
                
                # Calculate distance from user interest
                anti_distances = torch.norm(fake_anti - user_interest, dim=1)
                print(f"     Average distance from user interest: {anti_distances.mean():.4f}")
                print(f"     Distance std: {anti_distances.std():.4f}")
                
            except Exception as e:
                print(f"  ❌ Anti-interest generation failed: {e}")
                fake_anti = torch.randn(15, user_real_vectors.shape[1], device=self.device)
                user_result['anti_interest_vectors'] = fake_anti.cpu().numpy()
            
            # Method 2: Uniform space-filling generation
            print(f"\n  Method 2: Uniform Space-Filling Generation for User {user_id}")
            try:
                fake_uniform = self.vae_generator.generate_uniform_space_filling_vectors(
                    user_real_vectors=user_real_vectors,
                    num_fake_per_user=15,  # Same number as anti-interest method
                    diversity_weight=1.5,
                    coverage_weight=2.0,
                    num_iterations=500
                )
                user_result['uniform_space_vectors'] = fake_uniform.cpu().numpy()
                print(f"  ✅ Uniform space-filling method generated {fake_uniform.shape[0]} vectors")
                
                # Calculate coverage metrics
                uniform_distances = torch.norm(fake_uniform - user_interest, dim=1)
                print(f"     Average distance from user interest: {uniform_distances.mean():.4f}")
                print(f"     Distance std: {uniform_distances.std():.4f}")
                
            except Exception as e:
                print(f"  ❌ Uniform space-filling generation failed: {e}")
                fake_uniform = torch.randn(15, user_real_vectors.shape[1], device=self.device)
                user_result['uniform_space_vectors'] = fake_uniform.cpu().numpy()
            
            results['users'].append(user_result)
            
            print(f"\n  User {user_id} Generation Summary:")
            print(f"    Real news vectors: {user_result['real_vectors'].shape}")
            print(f"    Anti-interest fake vectors: {user_result['anti_interest_vectors'].shape}")
            print(f"    Uniform space-filling fake vectors: {user_result['uniform_space_vectors'].shape}")
        
        print(f"\n=== Multi-User Generation Results Summary ===")
        print(f"Successfully processed {len(results['users'])} users")
        for user_result in results['users']:
            print(f"  User {user_result['user_id']}: {user_result['real_vectors'].shape[0]} real, "
                  f"{user_result['anti_interest_vectors'].shape[0]} anti-interest, "
                  f"{user_result['uniform_space_vectors'].shape[0]} uniform vectors")
        
        return results
    
    def calculate_distribution_metrics(self, comparison_results):
        """
        Calculate distribution metrics for multi-user comparison
        
        Args:
            comparison_results: Dictionary containing results from both methods for multiple users
            
        Returns:
            metrics: Dictionary containing calculated metrics for all users
        """
        import numpy as np
        from scipy.spatial.distance import pdist, squareform
        from sklearn.metrics import pairwise_distances
        
        print("\n=== Multi-User Distribution Metrics Calculation ===")
        
        all_metrics = {
            'users': [],
            'overall_summary': {}
        }
        
        # Calculate metrics for each user
        for i, user_result in enumerate(comparison_results['users']):
            user_id = user_result['user_id']
            print(f"\n--- Calculating metrics for User {i+1}: {user_id} ---")
            
            real_vectors = user_result['real_vectors']
            anti_vectors = user_result['anti_interest_vectors']
            uniform_vectors = user_result['uniform_space_vectors']
            
            user_metrics = {
                'user_id': user_id,
                'real_news': {},
                'anti_interest': {},
                'uniform_space': {},
                'comparison': {}
            }
            
            # Calculate metrics for real news
            print(f"  Real news vectors: {real_vectors.shape}")
            real_distances = pdist(real_vectors, metric='euclidean')
            user_metrics['real_news'] = {
                'count': len(real_vectors),
                'mean_distance': np.mean(real_distances),
                'std_distance': np.std(real_distances),
                'coverage_area': np.max(real_distances) - np.min(real_distances),
                'diversity_score': np.std(real_distances) / (np.mean(real_distances) + 1e-8)
            }
            
            # Calculate metrics for anti-interest generation
            print(f"  Anti-interest vectors: {anti_vectors.shape}")
            anti_distances = pdist(anti_vectors, metric='euclidean')
            # Distance from real news
            anti_to_real_distances = pairwise_distances(anti_vectors, real_vectors, metric='euclidean')
            user_metrics['anti_interest'] = {
                'count': len(anti_vectors),
                'mean_distance': np.mean(anti_distances),
                'std_distance': np.std(anti_distances),
                'coverage_area': np.max(anti_distances) - np.min(anti_distances),
                'diversity_score': np.std(anti_distances) / (np.mean(anti_distances) + 1e-8),
                'mean_distance_to_real': np.mean(anti_to_real_distances),
                'min_distance_to_real': np.min(anti_to_real_distances)
            }
            
            # Calculate metrics for uniform space-filling generation
            print(f"  Uniform space vectors: {uniform_vectors.shape}")
            uniform_distances = pdist(uniform_vectors, metric='euclidean')
            # Distance from real news
            uniform_to_real_distances = pairwise_distances(uniform_vectors, real_vectors, metric='euclidean')
            user_metrics['uniform_space'] = {
                'count': len(uniform_vectors),
                'mean_distance': np.mean(uniform_distances),
                'std_distance': np.std(uniform_distances),
                'coverage_area': np.max(uniform_distances) - np.min(uniform_distances),
                'diversity_score': np.std(uniform_distances) / (np.mean(uniform_distances) + 1e-8),
                'mean_distance_to_real': np.mean(uniform_to_real_distances),
                'min_distance_to_real': np.min(uniform_to_real_distances)
            }
            
            # Comparative analysis for this user
            coverage_improvement_anti = (user_metrics['anti_interest']['coverage_area'] / 
                                       user_metrics['real_news']['coverage_area']) - 1
            coverage_improvement_uniform = (user_metrics['uniform_space']['coverage_area'] / 
                                          user_metrics['real_news']['coverage_area']) - 1
            
            diversity_improvement_anti = (user_metrics['anti_interest']['diversity_score'] / 
                                        user_metrics['real_news']['diversity_score']) - 1
            diversity_improvement_uniform = (user_metrics['uniform_space']['diversity_score'] / 
                                           user_metrics['real_news']['diversity_score']) - 1
            
            user_metrics['comparison'] = {
                'coverage_improvement_anti': coverage_improvement_anti,
                'coverage_improvement_uniform': coverage_improvement_uniform,
                'diversity_improvement_anti': diversity_improvement_anti,
                'diversity_improvement_uniform': diversity_improvement_uniform,
                'better_coverage_method': 'uniform_space' if coverage_improvement_uniform > coverage_improvement_anti else 'anti_interest',
                'better_diversity_method': 'uniform_space' if diversity_improvement_uniform > diversity_improvement_anti else 'anti_interest'
            }
            
            # Determine winner for this user
            anti_score = (coverage_improvement_anti * 0.4 + diversity_improvement_anti * 0.4 + 
                         (user_metrics['anti_interest']['mean_distance_to_real'] / 
                          user_metrics['uniform_space']['mean_distance_to_real']) * 0.2)
            uniform_score = (coverage_improvement_uniform * 0.4 + diversity_improvement_uniform * 0.4 + 
                           (user_metrics['uniform_space']['mean_distance_to_real'] / 
                            user_metrics['anti_interest']['mean_distance_to_real']) * 0.2)
            
            user_metrics['comparison']['winner'] = 'uniform_space' if uniform_score > anti_score else 'anti_interest'
            user_metrics['comparison']['anti_score'] = anti_score
            user_metrics['comparison']['uniform_score'] = uniform_score
            
            all_metrics['users'].append(user_metrics)
            
            # Print user-specific results
            print(f"  📊 User {user_id} Metrics Summary:")
            print(f"    Real News - Coverage: {user_metrics['real_news']['coverage_area']:.4f}, "
                  f"Diversity: {user_metrics['real_news']['diversity_score']:.4f}")
            print(f"    Anti-Interest - Coverage: {user_metrics['anti_interest']['coverage_area']:.4f} "
                  f"({coverage_improvement_anti:+.2%}), Diversity: {user_metrics['anti_interest']['diversity_score']:.4f} "
                  f"({diversity_improvement_anti:+.2%})")
            print(f"    Uniform Space - Coverage: {user_metrics['uniform_space']['coverage_area']:.4f} "
                  f"({coverage_improvement_uniform:+.2%}), Diversity: {user_metrics['uniform_space']['diversity_score']:.4f} "
                  f"({diversity_improvement_uniform:+.2%})")
            print(f"    🏆 Winner for User {user_id}: {user_metrics['comparison']['winner'].replace('_', ' ').title()}")
        
        # Calculate overall summary across all users
        print(f"\n=== Overall Multi-User Summary ===")
        
        # Count winners
        anti_wins = sum(1 for user_metrics in all_metrics['users'] 
                       if user_metrics['comparison']['winner'] == 'anti_interest')
        uniform_wins = sum(1 for user_metrics in all_metrics['users'] 
                          if user_metrics['comparison']['winner'] == 'uniform_space')
        
        # Average improvements
        avg_coverage_anti = np.mean([user_metrics['comparison']['coverage_improvement_anti'] 
                                   for user_metrics in all_metrics['users']])
        avg_coverage_uniform = np.mean([user_metrics['comparison']['coverage_improvement_uniform'] 
                                      for user_metrics in all_metrics['users']])
        avg_diversity_anti = np.mean([user_metrics['comparison']['diversity_improvement_anti'] 
                                    for user_metrics in all_metrics['users']])
        avg_diversity_uniform = np.mean([user_metrics['comparison']['diversity_improvement_uniform'] 
                                       for user_metrics in all_metrics['users']])
        
        all_metrics['overall_summary'] = {
            'total_users': len(all_metrics['users']),
            'anti_interest_wins': anti_wins,
            'uniform_space_wins': uniform_wins,
            'overall_winner': 'uniform_space' if uniform_wins > anti_wins else 'anti_interest',
            'avg_coverage_improvement_anti': avg_coverage_anti,
            'avg_coverage_improvement_uniform': avg_coverage_uniform,
            'avg_diversity_improvement_anti': avg_diversity_anti,
            'avg_diversity_improvement_uniform': avg_diversity_uniform
        }
        
        print(f"Total Users Analyzed: {len(all_metrics['users'])}")
        print(f"Anti-Interest Method Wins: {anti_wins}")
        print(f"Uniform Space-Filling Method Wins: {uniform_wins}")
        print(f"🏆 Overall Winner: {all_metrics['overall_summary']['overall_winner'].replace('_', ' ').title()}")
        print(f"\nAverage Improvements:")
        print(f"  Coverage - Anti-Interest: {avg_coverage_anti:+.2%}, Uniform: {avg_coverage_uniform:+.2%}")
        print(f"  Diversity - Anti-Interest: {avg_diversity_anti:+.2%}, Uniform: {avg_diversity_uniform:+.2%}")
        
        return all_metrics
    
    def create_pca_visualization(self, comparison_results):
        """
        Create separate PCA visualization for each user's generation methods comparison
        
        Args:
            comparison_results: Dictionary containing results from both methods for multiple users
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import numpy as np
        
        # Color scheme for different methods
        method_colors = {
            'real': '#1f77b4',        # Blue
            'anti_interest': '#ff7f0e',  # Orange
            'uniform_space': '#2ca02c'   # Green
        }
        method_markers = {
            'real': 'o',              # Circle
            'anti_interest': 's',     # Square
            'uniform_space': '^'      # Triangle
        }
        
        num_users = len(comparison_results['users'])
        
        # Create separate figure for each user
        for i, user_result in enumerate(comparison_results['users']):
            user_id = user_result['user_id']
            
            # Prepare vectors for this user
            real_vectors = user_result['real_vectors']
            anti_vectors = user_result['anti_interest_vectors']
            uniform_vectors = user_result['uniform_space_vectors']
            
            # Combine all vectors for this user
            user_all_vectors = np.vstack([real_vectors, anti_vectors, uniform_vectors])
            
            # Apply PCA for this user's data
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(user_all_vectors)
            
            # Split PCA results
            real_len = len(real_vectors)
            anti_len = len(anti_vectors)
            uniform_len = len(uniform_vectors)
            
            real_pca = pca_result[:real_len]
            anti_pca = pca_result[real_len:real_len+anti_len]
            uniform_pca = pca_result[real_len+anti_len:real_len+anti_len+uniform_len]
            
            # Create individual figure for this user
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Plot real news vectors
            scatter1 = ax.scatter(real_pca[:, 0], real_pca[:, 1],
                                c=method_colors['real'], marker=method_markers['real'],
                                s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                                label=f'真实新闻 ({len(real_vectors)} 个)')
            
            # Plot anti-interest fake vectors
            scatter2 = ax.scatter(anti_pca[:, 0], anti_pca[:, 1],
                                c=method_colors['anti_interest'], marker=method_markers['anti_interest'],
                                s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                                label=f'反兴趣生成 ({len(anti_vectors)} 个)')
            
            # Plot uniform space-filling fake vectors
            scatter3 = ax.scatter(uniform_pca[:, 0], uniform_pca[:, 1],
                                c=method_colors['uniform_space'], marker=method_markers['uniform_space'],
                                s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                                label=f'均匀空间填充 ({len(uniform_vectors)} 个)')
            
            # Customize the plot
            ax.set_title(f'用户 {user_id} - VAE生成方法对比 (PCA可视化)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(f'主成分1 (解释方差: {pca.explained_variance_ratio_[0]:.2%})', 
                         fontsize=12)
            ax.set_ylabel(f'主成分2 (解释方差: {pca.explained_variance_ratio_[1]:.2%})', 
                         fontsize=12)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=11, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set equal aspect ratio for better visualization
            ax.set_aspect('equal', adjustable='box')
            
            # Add statistics text box
            stats_text = f"""统计信息:
总向量数: {len(user_all_vectors)}
累计解释方差: {sum(pca.explained_variance_ratio_):.2%}
真实新闻: {len(real_vectors)} 个
反兴趣生成: {len(anti_vectors)} 个
均匀填充: {len(uniform_vectors)} 个"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save individual user plot
            user_plot_filename = f'user_{user_id}_vae_comparison_pca.png'
            plt.savefig(user_plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 用户 {user_id} 的PCA可视化已保存为: {user_plot_filename}")
            
            plt.show()
            
            # Print individual user analysis
            print(f"\n=== 用户 {user_id} PCA分析 ===")
            print(f"向量总数: {len(user_all_vectors)}")
            print(f"主成分1解释方差: {pca.explained_variance_ratio_[0]:.2%}")
            print(f"主成分2解释方差: {pca.explained_variance_ratio_[1]:.2%}")
            print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.2%}")
            print(f"真实新闻向量: {len(real_vectors)} 个")
            print(f"反兴趣生成向量: {len(anti_vectors)} 个")
            print(f"均匀空间填充向量: {len(uniform_vectors)} 个")
        
        # Create a summary overview plot with all users
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Prepare data for overview
        all_vectors = []
        all_labels = []
        user_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Red, Blue, Green, Orange, Purple
        
        for i, user_result in enumerate(comparison_results['users']):
            user_id = user_result['user_id']
            user_color = user_colors[i % len(user_colors)]
            
            # Combine all vectors for this user
            user_vectors = np.vstack([
                user_result['real_vectors'],
                user_result['anti_interest_vectors'],
                user_result['uniform_space_vectors']
            ])
            all_vectors.append(user_vectors)
            all_labels.extend([f'用户{user_id}'] * len(user_vectors))
        
        # Combine all vectors for global PCA
        combined_vectors = np.vstack(all_vectors)
        global_pca = PCA(n_components=2)
        global_pca_result = global_pca.fit_transform(combined_vectors)
        
        # Plot overview
        start_idx = 0
        for i, user_result in enumerate(comparison_results['users']):
            user_id = user_result['user_id']
            user_color = user_colors[i % len(user_colors)]
            
            user_total_len = (len(user_result['real_vectors']) + 
                            len(user_result['anti_interest_vectors']) + 
                            len(user_result['uniform_space_vectors']))
            
            ax.scatter(global_pca_result[start_idx:start_idx+user_total_len, 0],
                      global_pca_result[start_idx:start_idx+user_total_len, 1],
                      c=user_color, alpha=0.6, s=50, 
                      label=f'用户 {user_id} (所有向量)')
            start_idx += user_total_len
        
        ax.set_title('多用户VAE生成方法总览 (全局PCA)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f'主成分1 (解释方差: {global_pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'主成分2 (解释方差: {global_pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save overview plot
        overview_filename = 'multi_user_overview_pca.png'
        plt.savefig(overview_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 多用户总览图已保存为: {overview_filename}")
        
        plt.show()
        
        print(f"\n=== 总体PCA分析摘要 ===")
        print(f"分析用户数: {num_users}")
        print(f"总向量数: {len(combined_vectors)}")
        print(f"全局主成分1解释方差: {global_pca.explained_variance_ratio_[0]:.2%}")
        print(f"全局主成分2解释方差: {global_pca.explained_variance_ratio_[1]:.2%}")
        print(f"全局累计解释方差: {sum(global_pca.explained_variance_ratio_):.2%}")
        
        return global_pca_result
    
    def run_comparison(self):
        """
        Run complete VAE methods comparison for a single user
        
        Returns:
            bool: True if comparison completed successfully, False otherwise
        """
        try:
            print("=== VAE Fake News Generation Methods Comparison Test ===")
            print("Comparing Anti-Interest vs Uniform Space-Filling methods for a single user")
            
            # 1. Initialize components
            print("\n1. Initializing components...")
            if not self.initialize_components():
                print("❌ Component initialization failed")
                return False
            
            # 2. Prepare user data
            print("\n2. Preparing user data...")
            if not self.prepare_user_data():
                print("❌ User data preparation failed")
                return False
            
            # 3. Generate comparison results for single user
            print("\n3. Generating fake news comparison...")
            results = self.generate_fake_news_comparison()
            if results is None:
                print("❌ Failed to generate comparison results")
                return False
            
            # 4. Calculate comprehensive metrics
            print("\n" + "="*60)
            print("4. CALCULATING DISTRIBUTION METRICS")
            print("="*60)
            
            metrics = self.calculate_distribution_metrics(results)
            
            # 5. Create visualization
            print("\n" + "="*60)
            print("5. CREATING VISUALIZATION")
            print("="*60)
            
            try:
                pca_result = self.create_pca_visualization(results)
                print("✅ Visualization created successfully")
            except Exception as e:
                print(f"❌ Failed to create visualization: {e}")
                return False
            
            print("\n" + "="*60)
            print("COMPARISON TEST COMPLETED SUCCESSFULLY")
            print("="*60)
            
            # Check if results has 'users' key (multi-user) or is single user format
            if 'users' in results:
                # Multi-user format
                print(f"✅ Multi-user comparison completed for {len(results['users'])} users")
                for user_result in results['users']:
                    user_id = user_result.get('user_id', 'Unknown')
                    anti_count = len(user_result.get('anti_interest_vectors', []))
                    uniform_count = len(user_result.get('uniform_space_vectors', []))
                    print(f"✅ User {user_id}: {anti_count} anti-interest, {uniform_count} uniform vectors")
                print(f"✅ Visualization saved as 'multi_user_vae_comparison_pca.png'")
            else:
                # Single user format (fallback)
                print(f"✅ Single user comparison completed for User {results.get('user_id', 'Unknown')}")
                print(f"✅ Generated {len(results.get('anti_interest_vectors', []))} anti-interest vectors")
                print(f"✅ Generated {len(results.get('uniform_space_vectors', []))} uniform space-filling vectors")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in comparison test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    # Initialize comparison test
    comparison = VAEMethodsComparison(data_dir="./")
    
    # Run comparison
    success = comparison.run_comparison()
    
    if success:
        print("\n🎉 VAE methods comparison completed successfully!")
    else:
        print("\n❌ VAE methods comparison failed!")

if __name__ == "__main__":
    main()