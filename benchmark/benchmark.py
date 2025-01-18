import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForMaskedLM
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import json
from typing import * 
import math
def calculate_mean_scale(data_path: str = "data/Data.csv"):
    mean = 7.504182527098148
    scale = 2.1676455136517343
    return mean, scale

def get_hyperparams(data_path: str = "./output/hyperopt_results.json"):
    with open(data_path, 'r') as file:
        params = json.load(file)['best_params']  
    return tuple(params.values())
class ModelConfig:
    def __init__(self,
                 input_dim: int = 2560,
                 hyperparams_path: Optional[str] = None,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.random_state = random_state
        # Update default params to match checkpoint dimensions
        default_params = (384, 160, 4, 4, 0.1, 16, 6.30288565853412e-05)
        (self.embedding_dim,
         self.linear_dim,
         self.num_attention_layers,
         self.num_heads,
         self.dropout_rate,
         self.batch_size,
         self.learning_rate) = get_hyperparams(hyperparams_path) if hyperparams_path else default_params

class ProteinProteinAffinityLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.protein_projection = nn.Linear(
            config.input_dim,
            config.embedding_dim
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_attention_layers
        )
        
        self.affinity_head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.linear_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim, config.linear_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim // 2, 1)
        )

    def forward(self, protein1_embedding, protein2_embedding):
        protein1_proj = self.protein_projection(protein1_embedding)
        protein2_proj = self.protein_projection(protein2_embedding)
        combined = torch.stack([protein1_proj, protein2_proj], dim=1)
        transformed = self.transformer(combined)
        pooled = transformed.mean(dim=1)
        return self.affinity_head(pooled)

class ProteinProteinAffinityTrainer:
    def __init__(self, config=None, device=None, cache_dir=None):
        self.config = config or ModelConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ProteinProteinAffinityLM(self.config).to(self.device)
        self.ankh_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.ankh_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.ankh_model.eval()
        self.ankh_model.to(self.device)
        
    def encode_proteins(self, proteins: List[str], batch_size: int = 2) -> torch.Tensor:
        """Encode proteins using the ESM model with caching"""
        embeddings = []
        protein_cache = {}  # Initialize protein_cache as an empty dictionary
        
        for i in range(0, len(proteins), batch_size):
            batch = proteins[i:i+batch_size]
            batch_embeddings = []
            
            for protein_seq in batch:
                protein_seq = str(protein_seq).strip()
                cached_embedding = protein_cache.get(protein_seq)
                if cached_embedding is not None:
                    # Convert cached embedding to float32
                    batch_embeddings.append(cached_embedding.to(torch.float32))
                    continue
                
                # Use "esm2_t30_150M_UR50D" transformers AutoModelMaskedLM
                model = self.ankh_model
                tokenizer = self.ankh_tokenizer
                
                inputs = tokenizer(protein_seq, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    embedding = last_hidden_state.mean(dim=1)
                    protein_cache[protein_seq] = embedding.cpu()
                    batch_embeddings.append(embedding)
            
            embeddings.extend([emb.to(self.device) for emb in batch_embeddings])
        
        return torch.cat(embeddings)

    def evaluate(self, test_loader: DataLoader, data_path: str = "Data.csv") -> Dict[str, float]:
        """Evaluate the model on test data with proper denormalization"""
        self.model.eval()
        mean, scale = calculate_mean_scale(data_path)  # Get normalization parameters
        criterion = nn.MSELoss()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for p1_seqs, p2_seqs, affinities in tqdm(test_loader, desc='Evaluating'):
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                
                # Get normalized outputs
                outputs = self.model(p1_embeddings, p2_embeddings)
                outputs = outputs.squeeze()
                
                # Denormalize predictions (outputs) and actual values (affinities)
                denorm_outputs = outputs * scale + mean
                denorm_affinities = affinities * scale + mean
                
                # Calculate loss on denormalized values
                loss = criterion(denorm_outputs, denorm_affinities.to(self.device))
                total_loss += loss.item()
                
                # Store denormalized values
                predictions.extend(denorm_outputs.cpu().numpy())
                actuals.extend(denorm_affinities.cpu().numpy())
        
        mse = total_loss / len(test_loader)
        rmse = math.sqrt(mse)
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / len(actuals)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'actuals': actuals
        }

def calculate_metrics(actual, predicted):
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    r_value, p_value = stats.pearsonr(actual, predicted)
    r_squared = r2_score(actual, predicted)
    spearman_rho, spearman_p = stats.spearmanr(actual, predicted)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R_value': r_value,
        'R_squared': r_squared,
        'p_value': p_value,
        'Spearman_rho': spearman_rho,
        'Spearman_p': spearman_p
    }

def run_inference(input_csv, model_path='./models/protein_protein_affinity_esm_vs_ankh_best.pt', data_path='./data/Data.csv'):
    """Run inference using the trained model on protein pairs"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize trainer with config
    config = ModelConfig()
    trainer = ProteinProteinAffinityTrainer(config=config)
    
    # Load trained model weights
    checkpoint = torch.load(model_path, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    
    # Get normalization parameters
    mean, scale = calculate_mean_scale(data_path)
    
    # Read input data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} protein pairs from {input_csv}")
    
    results = []
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing protein pairs"):
            try:
                protein1_seq = row['protein1_sequence']
                protein2_seq = row['protein2_sequence']
                actual_pkd = row['pkd']
                
                # Get embeddings
                p1_embedding = trainer.encode_proteins([protein1_seq])
                p2_embedding = trainer.encode_proteins([protein2_seq])
                
                # Get normalized prediction
                normalized_pred = trainer.model(p1_embedding, p2_embedding).item()
                
                # Denormalize prediction
                prediction = normalized_pred * scale + mean
                
                results.append({
                    'Actual_pKd': actual_pkd,
                    'Predicted_pKd': prediction,
                    'Absolute_Error': abs(actual_pkd - prediction)
                })
                
            except Exception as e:
                logger.error(f"Error processing pair {index}: {str(e)}")
                continue
    
    # Convert results to DataFrame and calculate metrics
    results_df = pd.DataFrame(results)
    metrics = calculate_metrics(
        results_df['Actual_pKd'].values,
        results_df['Predicted_pKd'].values
    )
    
    # Print metrics
    logger.info("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")
    
    # Save results
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'binding_predictions_new.csv', index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / 'binding_metrics_new.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['Actual_pKd'], results_df['Predicted_pKd'], alpha=0.5)
    plt.plot([results_df['Actual_pKd'].min(), results_df['Actual_pKd'].max()], 
             [results_df['Actual_pKd'].min(), results_df['Actual_pKd'].max()], 
             'r--', label='Perfect prediction')
    plt.xlabel('Actual pKd')
    plt.ylabel('Predicted pKd')
    plt.title('Predicted vs Actual pKd Values')
    plt.legend()
    plt.savefig(output_dir / 'prediction_correlation_new.png')
    plt.close()
    
    return results_df, metrics

if __name__ == "__main__":
    # Run inference on benchmark data
    results, metrics = run_inference('benchmark.csv')
