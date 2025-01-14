import re
import torch
import ankh
import torch.nn as nn
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from Bio import SeqIO
import optuna
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
@dataclass
class ModelConfig:
    def __init__(self,
                 input_dim=2560,  # Match Ankh embedding dimension
                 embedding_dim=256,
                 linear_dim=128,
                 num_attention_layers=3,
                 num_heads=4,
                 dropout_rate=0.1):
        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

def calculate_mean_scale(data_path: str = "data/Data.csv"): 
    df = pd.read_csv(data_path, na_values=['Na'])  # Remove rows with NaN values
    affinities = df['pkd'].dropna()  # Remove rows with NaN values
    mean = affinities.mean()
    scale = affinities.std()
    return mean, scale

class ProteinPairDataset(Dataset):
    """Dataset for protein pairs and their binding affinities"""
    def __init__(self, protein1_sequences: List[str], 
                 protein2_sequences: List[str], 
                 affinities: torch.Tensor,
                 mean: float = None,
                 scale: float = None):
        assert len(protein1_sequences) == len(protein2_sequences) == len(affinities)
        self.protein1_sequences = [str(seq).strip() for seq in protein1_sequences]
        self.protein2_sequences = [str(seq).strip() for seq in protein2_sequences]
        
        # Calculate mean and scale if not provided
        mean, scale = calculate_mean_scale()
            
        # Normalize affinities
        self.affinities = (affinities -mean)/(scale)
        self.mean = mean
        self.scale = scale

    def __len__(self) -> int:
        return len(self.protein1_sequences)

    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (self.protein1_sequences[idx], 
                self.protein2_sequences[idx], 
                self.affinities[idx])

class ProteinProteinAffinityLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Project protein embeddings to model dimension
        self.protein_projection = nn.Linear(
            config.input_dim,
            config.embedding_dim
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_attention_layers
        )
        
        # Prediction head
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

    def forward(self, protein1_embedding: torch.Tensor, 
                protein2_embedding: torch.Tensor) -> torch.Tensor:
        protein1_proj = self.protein_projection(protein1_embedding)
        protein2_proj = self.protein_projection(protein2_embedding)
        combined = torch.stack([protein1_proj, protein2_proj], dim=1)
        transformed = self.transformer(combined)
        pooled = transformed.mean(dim=1)
        return self.affinity_head(pooled)

class ProteinEmbeddingCache:
    """Cache for storing protein embeddings to avoid recomputation"""
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache = {}
        # Set default cache directory if none provided
        self.cache_dir = Path(cache_dir if cache_dir else 'embedding_cache_2560')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track statistics
        self.total_added = 0
        self.total_loaded = 0
        self.save_counter = 0
        
        # Try to load existing cache from the specified location
        self.load('caches.pt')
        
        # Log initial cache state with more details
        logging.info(f"Cache initialized at {self.cache_dir}")
        logging.info(f"Initial cache size: {len(self.cache)} embeddings")
        if len(self.cache) > 0:
            sample_key = next(iter(self.cache))
            sample_embedding = self.cache[sample_key]
            logging.info(f"Sample embedding shape: {sample_embedding.shape}")
    def get(self, protein_sequence: str) -> Optional[torch.Tensor]:
        """Get embedding from cache with logging"""
        embedding = self.cache.get(protein_sequence)
        return embedding
    
    def set(self, protein_sequence: str, embedding: torch.Tensor):
        """Set embedding in cache with periodic saving"""
        self.cache[protein_sequence] = embedding
        self.total_added += 1
        
        # Log progress periodically
        if self.total_added % 500 == 0:
            logging.info(f"Added {self.total_added} embeddings to cache")
            logging.info(f"Current cache size: {len(self.cache)}")
        
        # Save periodically (every 1000 additions) to avoid too frequent disk writes
        if self.total_added % 1000 == 0:
            self.save('caches.pt')
            print("saved 1000 proteins")
    def save(self, filename: str):
        """Save cache to disk with detailed logging"""
        try:
            cache_path = self.cache_dir / filename
            self.save_counter += 1
            
            # Log save attempt
          
            # Create parent directories if they don't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cache dictionary
            torch.save(self.cache, cache_path)
            logging.info(f"Successfully saved cache to {cache_path}")
            
            # Verify save
            if cache_path.exists():
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                logging.info(f"Cache file size: {size_mb:.2f} MB")
        except Exception as e:
            logging.error(f"Failed to save cache to {cache_path}: {str(e)}", exc_info=True)
    
    def load(self, filename: str) -> bool:
        """Load cache from disk with detailed logging"""
        try:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                logging.info(f"Found existing cache file ({size_mb:.2f} MB)")
                
                self.cache = torch.load(cache_path)
                logging.info(f"Successfully loaded cache from {cache_path}")
                logging.info(f"Cache contains {len(self.cache)} protein embeddings")
                
                # Log some cache statistics
                if len(self.cache) > 0:
                    sample_key = next(iter(self.cache))
                    sample_embedding = self.cache[sample_key]
                    logging.info(f"Sample protein length: {len(sample_key)}")
                    logging.info(f"Sample embedding shape: {sample_embedding.shape}")
                return True
            else:
                logging.info(f"No existing cache found at {cache_path}")
                return False
        except Exception as e:
            logging.error(f"Failed to load cache from {cache_path}: {str(e)}", exc_info=True)
            return False

    def __len__(self) -> int:
        """Return number of cached embeddings"""
        return len(self.cache)

class ProteinProteinAffinityTrainer:
    def __init__(self, 
                 config: Optional[ModelConfig] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.config = config or ModelConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model = ProteinProteinAffinityLM(self.config).to(self.device)
        self.ankh_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.ankh_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t36_3B_UR50D")
        self.ankh_model.eval()
        self.ankh_model.to(self.device)
        
        # Initialize embedding cache with specified or default cache directory
        self.protein_cache = ProteinEmbeddingCache(cache_dir)

    def encode_proteins(self, proteins: List[str], batch_size: int = 2) -> torch.Tensor:
        """Encode proteins using the Ankh model with caching"""
        embeddings = []
        
        for i in range(0, len(proteins), batch_size):
            batch = proteins[i:i+batch_size]
            batch_embeddings = []
            
            for protein in batch:
                protein = str(protein).strip()
                cached_embedding = self.protein_cache.get(protein)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    continue
                
                tokens = self.ankh_tokenizer([protein], 
                                          padding=True, 
                                          return_tensors="pt")
                with torch.no_grad():
                    output = self.ankh_model(
                        input_ids=tokens['input_ids'].to(self.device),
                        attention_mask=tokens['attention_mask'].to(self.device),
                        output_hidden_states=True
                    )
                    # Get the last hidden state from the output tuple
                    last_hidden_state = output.hidden_states[-1]
                    embedding = last_hidden_state.mean(dim=1)
                    self.protein_cache.set(protein, embedding.cpu())
                    batch_embeddings.append(embedding)
            
            embeddings.extend([emb.to(self.device) for emb in batch_embeddings])
        
        return torch.cat(embeddings)

    def prepare_data(self, 
                    protein1_sequences: List[str],
                    protein2_sequences: List[str],
                    affinities: List[float],
                    benchmark_path: str = "benchmark.csv",
                    batch_size: int = 32,
                    test_size: float = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test data loaders"""
        affinities_tensor = torch.tensor(affinities, dtype=torch.float32)
        
        # Load benchmark data for validation
        try:
            benchmark_df = pd.read_csv(benchmark_path, na_values=['Na'])
            val_p1 = benchmark_df['protein1_sequence'].tolist()
            val_p2 = benchmark_df['protein2_sequence'].tolist()
            val_aff = torch.tensor(benchmark_df['pkd'].tolist(), dtype=torch.float32)
            
            logging.info(f"Loaded {len(val_p1)} protein pairs from benchmark dataset")
        except Exception as e:
            logging.error(f"Failed to load benchmark data: {str(e)}")
            raise
            
        # When test_size is 0, use all data for training
        if test_size == 0:
            train_p1, train_p2, train_aff = protein1_sequences, protein2_sequences, affinities_tensor
            test_p1, test_p2, test_aff = [], [], torch.tensor([])
        else:
            # Split remaining data into train and test
            train_p1, test_p1, train_p2, test_p2, train_aff, test_aff = train_test_split(
                protein1_sequences, protein2_sequences, affinities_tensor,
                test_size=test_size, random_state=42
            )
        train_dataset = ProteinPairDataset(train_p1, train_p2, train_aff)
        val_dataset = ProteinPairDataset(val_p1, val_p2, val_aff)
        test_dataset = ProteinPairDataset(test_p1, test_p2, test_aff)
        
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size),
            DataLoader(test_dataset, batch_size=batch_size)
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Define hyperparameter space
        config = ModelConfig(
            input_dim=2560,  # Fixed for Ankh
            embedding_dim=trial.suggest_int('embedding_dim', 128, 512, step=64),
            linear_dim=trial.suggest_int('linear_dim', 64, 256, step=32),
            num_attention_layers=trial.suggest_int('num_attention_layers', 2, 6),
            num_heads=trial.suggest_categorical('num_heads', [4, 8, 16]),
            dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        )
        
        # Training hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        
        # Initialize model with trial config
        self.model = ProteinProteinAffinityLM(config).to(self.device)
        
        # Prepare data with trial batch size
        train_loader, val_loader, _ = self.prepare_data(
            self.protein1_sequences,
            self.protein2_sequences,
            self.affinities,
            batch_size=batch_size
        )
        
        # Train model
        history = self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=150,  # Reduced for optimization
            learning_rate=learning_rate,
            patience=10
        )
        
        return min(history['val_loss'])  # Return best validation loss

    def train_with_hyperopt(self,
                           protein1_sequences: List[str],
                           protein2_sequences: List[str],
                           affinities: List[float],
                           n_trials: int = 30) -> Dict:
        """Train model with hyperparameter optimization"""
        self.protein1_sequences = protein1_sequences
        self.protein2_sequences = protein2_sequences
        self.affinities = affinities
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_params = study.best_params
        best_config = ModelConfig(
            input_dim=2560,
            embedding_dim=best_params['embedding_dim'],
            linear_dim=best_params['linear_dim'],
            num_attention_layers=best_params['num_attention_layers'],
            num_heads=best_params['num_heads'],
            dropout_rate=best_params['dropout_rate']
        )
        # Train final model with best hyperparameters
        self.model = ProteinProteinAffinityLM(best_config).to(self.device)
        train_loader, val_loader, test_loader = self.prepare_data(
            protein1_sequences,
            protein2_sequences,
            affinities,
            batch_size=best_params['batch_size']
        )
        
        history = self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=best_params['learning_rate']
        )
        
        return {
            'best_params': best_params,
            'best_val_loss': study.best_value,
            'history': history,
            'study': study
        }

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 100,
             learning_rate: float = 1e-4,
             save_dir: str = 'models',
             model_name: str = 'protein_protein_affinity_esm_vs_ankh_best.pt',
             patience: int = 10) -> Dict[str, List[float]]:
        save_path = Path(save_dir) / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            val_loss = self._validate_epoch(val_loader, criterion)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, save_path)
                print(f'Saved new best model with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        return history

    def _train_epoch(self, 
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for p1_seqs, p2_seqs, affinities in pbar:
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(p1_embeddings, p2_embeddings)
                outputs = outputs.squeeze()
                loss = criterion(outputs, affinities)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)

    def _validate_epoch(self, 
                       val_loader: DataLoader,
                       criterion: nn.Module) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for p1_seqs, p2_seqs, affinities in val_loader:
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                outputs = self.model(p1_embeddings, p2_embeddings)
                outputs = outputs.squeeze()
                loss = criterion(outputs, affinities)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data"""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for p1_seqs, p2_seqs, affinities in tqdm(test_loader, desc='Evaluating'):
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                outputs = self.model(p1_embeddings, p2_embeddings)
                outputs = outputs.squeeze()
                loss = criterion(outputs, affinities)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(affinities.cpu().numpy())
        
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

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('protein_affinity.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Create directories for outputs
        output_dir = Path('output')
        print(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        # Load and process data
        data_path = "data/Data.csv"
        logger.info(f"Loading data from {data_path}")

        
        protein1_sequences = pd.read_csv(data_path, na_values=['Na'])['protein1_sequence'].tolist()
        protein2_sequences = pd.read_csv(data_path, na_values=['Na'])['protein2_sequence'].tolist()
        affinities = pd.read_csv(data_path, na_values=['Na'])['pkd'].tolist()

        # Remove empty sequences
        valid_indices = [i for i in range(len(protein1_sequences))
                        if protein1_sequences[i] and protein2_sequences[i]]
        protein1_sequences = [protein1_sequences[i] for i in valid_indices]
        protein2_sequences = [protein2_sequences[i] for i in valid_indices]
        affinities = [affinities[i] for i in valid_indices]

        logger.info(f"Loaded {len(protein1_sequences)} protein pairs")

        # Initialize trainer
        logger.info("Initializing trainer with hyperparameter optimization...")
        trainer = ProteinProteinAffinityTrainer(
            cache_dir=str('embedding_cache_2560')
        )

        # Perform hyperparameter optimization
        logger.info("Starting hyperparameter optimization...")
        opt_results = trainer.train_with_hyperopt(
            protein1_sequences=protein1_sequences,
            protein2_sequences=protein2_sequences,
            affinities=affinities,
            n_trials=100
        )

        # Save optimization results
        with open(output_dir / 'hyperopt_results.json', 'w') as f:
            results_dict = {
                'best_params': opt_results['best_params'],
                'best_val_loss': float(opt_results['best_val_loss'])
            }
            json.dump(results_dict, f, indent=4)

        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.plot_optimization_history(opt_results['study'])
        plt.savefig(output_dir / 'optimization_history.png')
        plt.close()

        # Plot training history for final model
        plt.figure(figsize=(10, 6))
        history = opt_results['history']
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History (Best Hyperparameters)')
        plt.legend()
        plt.savefig(output_dir / 'final_training_history.png')
        plt.close()

        # Final evaluation
        _, val_loader, _ = trainer.prepare_data(
            protein1_sequences=protein1_sequences,
            protein2_sequences=protein2_sequences,
            affinities=affinities,
            batch_size=opt_results['best_params']['batch_size']
        )

        results = trainer.evaluate(val_loader)
        logger.info(f"Final test results: MSE={results['mse']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")

        # Save final evaluation results
        with open(output_dir / 'final_evaluation_results.json', 'w') as f:
            eval_results = {
                'mse': float(results['mse']),
                'rmse': float(results['rmse']),
                'mae': float(results['mae'])
            }
            json.dump(eval_results, f, indent=4)

        # Plot final predictions vs actuals
        plt.figure(figsize=(8, 8))
        plt.scatter(results['actuals'], results['predictions'], alpha=0.5)
        plt.plot([min(results['actuals']), max(results['actuals'])],
                [min(results['actuals']), max(results['actuals'])],
                'r--', label='Perfect Prediction')
        plt.xlabel('Actual Affinity')
        plt.ylabel('Predicted Affinity')
        plt.title('Predictions vs Actuals (Best Model)')
        plt.legend()
        plt.savefig(output_dir / 'final_predictions_vs_actuals.png')
        plt.close()

        logger.info(f"All outputs saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()


