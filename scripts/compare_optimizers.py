import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import numpy as np
import argparse
import glob
from typing import List, Dict, Any, Optional
from config.config import default_config as config
from utils.visualize import setup_visualization_dir, plot_seaborn_style, visualize_gradient_norms

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare optimizer performance for transformer training")
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="models",
        help="Directory containing model checkpoints from different optimizers"
    )
    parser.add_argument(
        "--optimizers", 
        type=str, 
        nargs='+',
        default=["ADAMW", "SGD", "ADAM", "GGD", "GGD_LW"],
        help="List of optimizers to compare"
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="transformer_tiny",
        help="Prefix of model checkpoint files"
    )
    return parser.parse_args()

def load_metrics_from_checkpoints(models_dir: str, optimizers: List[str], prefix: str) -> Dict[str, Any]:
    """
    Load training metrics from checkpoints.
    
    Args:
        models_dir: Directory containing model checkpoints
        optimizers: List of optimizer names to compare
        prefix: Prefix of model checkpoint files
        
    Returns:
        Dictionary containing metrics for each optimizer
    """
    metrics = {
        'train_losses': {},
        'dev_losses': {},
        'steps': {},
        'grad_norms': {},
        'layer_names': {}
    }
    
    # Load data for each optimizer
    for opt in optimizers:
        # Try to find both regular and best checkpoints
        checkpoint_patterns = [
            os.path.join(models_dir, f"{prefix}_{opt}.pt"),
            os.path.join(models_dir, f"{prefix}_best_{opt}.pt")
        ]
        
        for pattern in checkpoint_patterns:
            matching_files = glob.glob(pattern)
            if matching_files:
                checkpoint_path = matching_files[0]
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                    
                    # Extract metrics from checkpoint
                    if 'training_metrics' in checkpoint:
                        train_metrics = checkpoint['training_metrics']
                        if 'eval_train_losses' in train_metrics and train_metrics['eval_train_losses']:
                            metrics['train_losses'][opt] = train_metrics['eval_train_losses']
                            metrics['dev_losses'][opt] = train_metrics['eval_dev_losses']
                            metrics['steps'][opt] = train_metrics['eval_steps']
                        
                        # Extract gradient norm data if available
                        if 'gradient_norms' in train_metrics:
                            metrics['grad_norms'][opt] = train_metrics['gradient_norms']
                            metrics['layer_names'][opt] = train_metrics['layer_names']
                    else:
                        # Fall back to simpler metrics
                        if 'train_loss' in checkpoint:
                            metrics['train_losses'][opt] = [checkpoint['train_loss']]
                            metrics['dev_losses'][opt] = [checkpoint['dev_loss']]
                            metrics['steps'][opt] = [checkpoint.get('steps', 0)]
                    
                    print(f"Loaded metrics from {checkpoint_path}")
                except Exception as e:
                    print(f"Error loading checkpoint {checkpoint_path}: {e}")
    
    return metrics

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load metrics from checkpoints
    metrics = load_metrics_from_checkpoints(args.models_dir, args.optimizers, args.prefix)
    
    # Create visualization directory
    visuals_dir = setup_visualization_dir(args.models_dir)
    
    # 1. Compare training losses
    if metrics['train_losses']:
        plot_seaborn_style(
            metrics['train_losses'],
            range(max([len(losses) for losses in metrics['train_losses'].values()])),
            "Training Loss Comparison",
            "training_loss_comparison",
            "Loss",
            visuals_dir,
            xlabel="Evaluation Step"
        )
    
    # 2. Compare validation losses
    if metrics['dev_losses']:
        plot_seaborn_style(
            metrics['dev_losses'],
            range(max([len(losses) for losses in metrics['dev_losses'].values()])),
            "Validation Loss Comparison",
            "validation_loss_comparison",
            "Loss",
            visuals_dir,
            xlabel="Evaluation Step"
        )
    
    # 3. Compare gradient norms if available
    if metrics['grad_norms']:
        visualize_gradient_norms(
            metrics['grad_norms'],
            metrics['layer_names'],
            visuals_dir,
            "Gradient Norm Comparison"
        )
    
    print(f"Comparison visualizations saved to {visuals_dir}")

if __name__ == "__main__":
    main()
