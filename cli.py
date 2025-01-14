import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import r2_score
from index import ProteinProteinAffinityTrainer, ProteinProteinAffinityLM
def parse_args():
    parser = argparse.ArgumentParser(description='Protein-Protein Binding Affinity Prediction')
    
    # Input group - either CSV or raw sequences
    input_group = parser.add_mutually_exclusive_group(required=True)
    
    # CSV input option
    input_group.add_argument('--input_csv', type=str,
                          help='Path to CSV file containing protein sequences. Must have columns "protein1" and "protein2"')
    
    # Raw sequence input option
    input_group.add_argument('--sequences', nargs='+', action='append',
                          help='Raw protein sequence pairs. Pass pairs as separate arguments, e.g., --sequences seq1A seq1B --sequences seq2A seq2B')
    
    # Column names for CSV input
    parser.add_argument('--protein1_col', type=str, default='protein1_sequence',
                      help='Column name for first protein sequences in CSV (default: protein1)')
    parser.add_argument('--protein2_col', type=str, default='protein2_sequence',
                      help='Column name for second protein sequences in CSV (default: protein2)')
    
    # Optional arguments with defaults
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results (default: output)')
    parser.add_argument('--model_path', type=str, 
                      default='models/protein_protein_affinity_esm_vs_ankh_best.pt',
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='Data.csv',
                      help='Path to training data for normalization parameters')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run inference on (default: cuda if available, else cpu)')
    
    # Model configuration arguments
    parser.add_argument('--input_dim', type=int, default=2560,
                      help='Input dimension of protein embeddings')
    parser.add_argument('--embedding_dim', type=int, default=384,
                      help='Embedding dimension for transformer')
    parser.add_argument('--linear_dim', type=int, default=160,
                      help='Linear layer dimension')
    parser.add_argument('--num_attention_layers', type=int, default=2,
                      help='Number of transformer attention layers')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                      help='Dropout rate')
    
    return parser.parse_args()

def load_sequences_from_csv(file_path: str, protein1_col: str, protein2_col: str) -> Tuple[List[str], List[str]]:
    """Load protein sequences from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if protein1_col not in df.columns or protein2_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{protein1_col}' and '{protein2_col}'")
        
        sequences1 = df[protein1_col].tolist()
        sequences2 = df[protein2_col].tolist()
        
        return sequences1, sequences2
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def parse_raw_sequences(sequences: List[List[str]]) -> Tuple[List[str], List[str]]:
    """Parse raw sequence pairs from command line arguments."""
    if not sequences or not all(len(pair) == 2 for pair in sequences):
        raise ValueError("Raw sequences must be provided in pairs")
    
    sequences1, sequences2 = zip(*sequences)
    return list(sequences1), list(sequences2)

def calculate_mean_scale(data_path: str) -> Tuple[float, float]:
    """Calculate normalization parameters from training data."""
    df = pd.read_csv(data_path, na_values=['Na'])
    affinities = df['pkd'].dropna()
    return float(affinities.mean()), float(affinities.std())

def get_hyperparams(data_path: str = "output/hyperopt_results.json"):
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


def run_batch_inference(
    protein_sequences1: List[str],
    protein_sequences2: List[str],
    trainer,
    batch_size: int,
    mean: float,
    scale: float
) -> pd.DataFrame:
    """Run inference in batches."""
    results = []
    
    for i in tqdm(range(0, len(protein_sequences1), batch_size), desc="Processing batches"):
        batch_seq1 = protein_sequences1[i:i + batch_size]
        batch_seq2 = protein_sequences2[i:i + batch_size]
        
        # Get embeddings
        with torch.no_grad():
            p1_embeddings = trainer.encode_proteins(batch_seq1)
            p2_embeddings = trainer.encode_proteins(batch_seq2)
            
            # Get normalized predictions
            normalized_preds = trainer.model(p1_embeddings, p2_embeddings).squeeze()
            
            # Handle single prediction case
            if len(batch_seq1) == 1:
                normalized_preds = normalized_preds.unsqueeze(0)
                
            # Denormalize predictions
            predictions = (normalized_preds * scale + mean).cpu().numpy()
            
            for seq1, seq2, pred in zip(batch_seq1, batch_seq2, predictions):
                results.append({
                    'Protein1_Sequence': seq1,
                    'Protein2_Sequence': seq2,
                    'Predicted_pKd': float(pred)
                })
    
    return pd.DataFrame(results)

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load sequences based on input type
    try:
        if args.input_csv:
            logger.info(f"Loading sequences from CSV file: {args.input_csv}")
            sequences1, sequences2 = load_sequences_from_csv(
                args.input_csv,
                args.protein1_col,
                args.protein2_col
            )
        else:
            logger.info("Processing raw sequence pairs")
            sequences1, sequences2 = parse_raw_sequences(args.sequences)
        
        # Validate sequences
        logger.info(f"Successfully loaded {len(sequences1)} protein pairs")
        
    except Exception as e:
        logger.error(f"Error loading sequences: {str(e)}")
        return
    
    # Initialize model and trainer
    config = ModelConfig()
    trainer = ProteinProteinAffinityTrainer(config=config, device=device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    
    # Get normalization parameters
    mean, scale = calculate_mean_scale(args.data_path)
    logger.info(f"Normalization parameters - Mean: {mean:.4f}, Scale: {scale:.4f}")
    
    # Run inference
    try:
        results_df = run_batch_inference(
            sequences1,
            sequences2,
            trainer,
            args.batch_size,
            mean,
            scale
        )
        
        # Save results
        output_file = output_dir / 'binding_predictions.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        print(results_df)        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")

if __name__ == "__main__":
    main()
