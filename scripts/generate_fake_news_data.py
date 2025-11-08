#!/usr/bin/env python
import sys
import os
from pathlib import Path
import pickle
import torch
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataload.vae_fake_news_generator import IntegratedFakeNewsGenerator
from src.dataload.data_preprocess import read_news_behaviors # Assuming this function is accessible

# Mock cfg for now, will need to be properly loaded
class MockCfg:
    def __init__(self):
        self.dataset = MockDatasetCfg()
        self.model = MockModelCfg()
        self.path = MockPathCfg()
        self.gpu_num = 1 # Assuming single GPU for generation
        self.vae_latent_dim = 50 # Default value
        self.vae_training_epochs = 1000 # Default value
        self.fake_news_per_user = 20 # Default value
        self.vae_generation_iterations = 500 # Default value

class MockDatasetCfg:
    def __init__(self):
        self.dataset_lang = 'chinese' # Or 'english' based on your data
        self.data_dir = './data/MINDsmall' # Placeholder

class MockModelCfg:
    def __init__(self):
        self.his_size = 70 # Placeholder
        self.word_filter_num = 5 # Placeholder

class MockPathCfg:
    def __init__(self):
        self.checkpoint_dir = './checkpoint' # Placeholder

def generate_all_fake_news(cfg):
    print("Starting fake news generation...")

    # 1. Initialize IntegratedFakeNewsGenerator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_news_generator = IntegratedFakeNewsGenerator(cfg, device=device)
    
    # Initialize VAE (loads real news data and trains VAE)
    # This will populate fake_news_generator.news_vectors with real news vectors
    data_dir_path = Path(cfg.dataset.data_dir)
    if not fake_news_generator.initialize_vae(data_dir=str(data_dir_path)):
        print("Failed to initialize VAE. Exiting.")
        return

    # 2. Collect user histories (mimic data_preprocess.py)
    # This part needs to be adapted based on how your actual user behaviors are loaded
    # For now, let's assume we can get user_histories from a mock or actual source
    # You might need to adjust the path to your behavior file
    behavior_file_path = data_dir_path / "train" / "behaviors.tsv"
    if not behavior_file_path.exists():
        print(f"Error: Behavior file not found at {behavior_file_path}. Please update cfg.dataset.data_dir or behavior_file_path.")
        return

    user_histories = {}
    print("Collecting user histories...")
    with open(behavior_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Collecting user history"):
            parts = line.strip().split('\t')
            if len(parts) < 4: # Ensure line has enough parts
                continue
            uid = parts[1] # User ID is typically the second column
            history = parts[3] # History is typically the fourth column
            if uid not in user_histories:
                user_histories[uid] = []
            if history.strip():
                user_histories[uid].extend(history.split())
    
    # Deduplicate user histories
    for uid in user_histories:
        user_histories[uid] = list(set(user_histories[uid]))
    
    print(f"Collected histories for {len(user_histories)} users.")

    all_generated_fake_news_vectors = {}
    all_generated_fake_news_ids = []

    # 3. Generate personalized fake news for each user
    print("Generating personalized fake news for each user...")
    for uid, history_news_ids in tqdm(user_histories.items(), desc="Generating fake news"):
        user_history_vectors = []
        if history_news_ids:
            # Retrieve vectors for historical news from news_vectors (populated by initialize_vae)
            for news_id in history_news_ids:
                if news_id in fake_news_generator.news_vectors:
                    user_history_vectors.append(fake_news_generator.news_vectors[news_id])
        
        if user_history_vectors:
            user_history_vectors_tensor = torch.stack(user_history_vectors).to(device)
        else:
            user_history_vectors_tensor = None # No history, generate non-personalized

        # Call the method to generate fake news vectors
        # This method now returns IDs and implicitly stores vectors in fake_news_generator.news_vectors
        generated_ids = fake_news_generator.generate_personalized_fake_news_vectors(
            num_fake_news=cfg.fake_news_per_user,
            user_id=uid,
            user_history_vectors=user_history_vectors_tensor
        )
        
        # Collect the newly generated fake news vectors and IDs
        for fake_id in generated_ids:
            if fake_id in fake_news_generator.news_vectors:
                all_generated_fake_news_vectors[fake_id] = fake_news_generator.news_vectors[fake_id].cpu().numpy()
                all_generated_fake_news_ids.append(fake_id)

    # 4. Save all generated fake news data to disk
    output_dir = data_dir_path / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    fake_vectors_path = output_dir / "fake_news_vectors.bin"
    fake_ids_path = output_dir / "fake_news_ids.bin"

    print(f"Saving {len(all_generated_fake_news_ids)} fake news vectors to {fake_vectors_path}")
    with open(fake_vectors_path, 'wb') as f:
        pickle.dump(all_generated_fake_news_vectors, f)

    print(f"Saving {len(all_generated_fake_news_ids)} fake news IDs to {fake_ids_path}")
    with open(fake_ids_path, 'wb') as f:
        pickle.dump(all_generated_fake_news_ids, f)

    print("Fake news generation and saving completed.")

if __name__ == "__main__":
    # You might want to load a proper Hydra config here
    # For now, using a mock config
    mock_cfg = MockCfg()
    generate_all_fake_news(mock_cfg)