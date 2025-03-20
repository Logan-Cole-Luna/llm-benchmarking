import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import time
import argparse
from typing import Dict, List, Any
import copy
import numpy as np
from tqdm import tqdm
from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from GGD import GGD
from utils.visualize import setup_visualization_dir, plot_seaborn_style, visualize_gradient_norms, save_experiment_results, GradientTracker

def train_with_optimizer(optimizer_name: str) -> Dict[str, Any]:
    """
    Train a transformer model using the specified optimizer.
    
    Args:
        optimizer_name: Name of the optimizer to use
        
    Returns:
        Dictionary containing training metrics
    """
    print(f"\n{'='*80}\nTraining with optimizer: {optimizer_name}\n{'='*80}")
    
    # Initialize the model
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    ).to(config['device'])
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")
    
    # Initialize metrics tracking
    metrics_data = {
        'train_losses': [],
        'dev_losses': [],
        'eval_steps': [],
        'wall_times': [],
        'iteration_losses': []
    }
    
    # Initialize gradient tracker
    gradient_tracker = GradientTracker()
    
    # Setup optimizer
    optimizer_params = config['optimizer_params'][optimizer_name]
    
    if optimizer_name == "GGD" or optimizer_name == "GGD_LW":
        optimizer = GGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == "ADAMW":
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == "GGD_HYBRID":
        optimizer = GGD(model.parameters(), **optimizer_params)
    elif optimizer_name == "GGD_ADAM":
        # Renamed for clarity but functionally equivalent to previous implementation
        optimizer = GGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Using optimizer: {optimizer_name} with parameters: {optimizer_params}")
    
    # Training tracking
    losses = []
    AVG_WINDOW = 64
    
    # Helper function to estimate loss
    @torch.no_grad()
    def estimate_loss(steps: int):
        out = {}
        model.eval()
        
        for split in ['train', 'dev']:
            data_path = config['train_path'] if split == 'train' else config['dev_path']
            batch_iterator_eval = get_batch_iterator(
                data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
            )
            
            losses_eval = torch.zeros(steps)
            for k in range(steps):
                try:
                    xb, yb = next(batch_iterator_eval)
                    _, loss = model(xb, yb)
                    losses_eval[k] = loss.item()
                except StopIteration:
                    print(f"Warning: Iterator for {split} ended early.")
                    break
            
            out[split] = losses_eval[:k + 1].mean()
        
        model.train()
        return out
    
    # Training loop
    batch_iterator = get_batch_iterator(
        config['train_path'],
        config['t_batch_size'],
        config['t_context_length'],
        device=config['device']
    )
    
    # Track start time for walltime tracking
    start_time = time.time()
    
    # Training loop with progress bar
    pbar = tqdm(range(config['t_train_steps']))
    for step in pbar:
        try:
            # Get batch
            xb, yb = next(batch_iterator)
            
            # Forward pass
            _, loss = model(xb, yb)
            
            # Record metrics
            losses.append(loss.item())
            metrics_data['iteration_losses'].append(loss.item())
            metrics_data['wall_times'].append(time.time() - start_time)
            
            pbar.set_description(f"{optimizer_name} loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")
            
            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Track gradients
            gradient_tracker.update(model)
            
            optimizer.step()
            
            # Evaluation
            if step % config['t_eval_steps'] == 0:
                evaluation_losses = estimate_loss(config['t_eval_iters'])
                train_loss = evaluation_losses['train']
                dev_loss = evaluation_losses['dev']
                print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
                
                metrics_data['train_losses'].append(train_loss)
                metrics_data['dev_losses'].append(dev_loss)
                metrics_data['eval_steps'].append(step)
            
            # Learning rate decay
            if step == config['t_lr_decay_step']:
                print('Decaying learning rate')
                for g in optimizer.param_groups:
                    g['lr'] = config['t_lr_decayed']
        
        except StopIteration:
            print("Training data iterator finished early.")
            break
    
    # Final evaluation
    evaluation_losses = estimate_loss(config['t_eval_iters'])
    train_loss = evaluation_losses['train']
    dev_loss = evaluation_losses['dev']
    
    # Save model
    optimizer_out_path = config['t_out_path'].replace('.pt', f'_{optimizer_name}.pt')
    os.makedirs(os.path.dirname(optimizer_out_path), exist_ok=True)
    
    # Get gradient norm data
    gradient_norms, layer_names = gradient_tracker.get_data()
    
    # Save model with training metrics
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'steps': len(losses),
            'training_metrics': {
                'optimizer': optimizer_name,
                'eval_train_losses': metrics_data['train_losses'],
                'eval_dev_losses': metrics_data['dev_losses'],
                'eval_steps': metrics_data['eval_steps'],
                'iteration_losses': metrics_data['iteration_losses'],
                'wall_times': metrics_data['wall_times'],
                'gradient_norms': gradient_norms,
                'layer_names': layer_names,
            },
        },
        optimizer_out_path
    )
    
    print(f"Saved model to {optimizer_out_path}")
    print(f"Final metrics - Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
    
    return {
        'model': model,
        'optimizer_name': optimizer_name,
        'train_losses': metrics_data['train_losses'], 
        'dev_losses': metrics_data['dev_losses'],
        'eval_steps': metrics_data['eval_steps'],
        'iteration_losses': metrics_data['iteration_losses'],
        'wall_times': metrics_data['wall_times'],
        'gradient_norms': gradient_norms,
        'layer_names': layer_names,
    }

def create_comparative_visualizations(all_results: Dict[str, Dict[str, Any]], visuals_dir: str) -> None:
    """
    Create comparative visualizations for all trained optimizers.
    
    Args:
        all_results: Dictionary mapping optimizer names to their training results
        visuals_dir: Directory to save visualizations
    """
    # Ensure the visualization directory exists
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Helper function to convert tensors to lists safely
    def safe_convert(x):
        if isinstance(x, torch.Tensor):
            if x.dim() == 0:  # Handle 0-dim tensor
                return [x.item()]
            return x.detach().cpu().tolist()
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (list, tuple)):
            return list(x)
        else:
            return [x]  # Convert scalar to list
    
    # Process data for plotting
    train_losses = {}
    dev_losses = {}
    iteration_losses = {}
    wall_times = {}
    grad_norms = {}
    layer_names = {}
    
    # Extract and convert data
    for opt, results in all_results.items():
        # Skip optimizer if no results
        if not results:
            continue
            
        # Handle train/dev losses
        if 'train_losses' in results and results['train_losses']:
            train_losses[opt] = safe_convert(results['train_losses'])
        
        if 'dev_losses' in results and results['dev_losses']:
            dev_losses[opt] = safe_convert(results['dev_losses'])
        
        # Handle iteration losses
        if 'iteration_losses' in results and results['iteration_losses']:
            iteration_losses[opt] = safe_convert(results['iteration_losses'])
            
        # Handle wall times
        if 'wall_times' in results and results['wall_times']:
            wall_times[opt] = safe_convert(results['wall_times'])
            
        # Handle gradient data
        if 'gradient_norms' in results and results['gradient_norms']:
            grad_norms[opt] = results['gradient_norms']  # This is a list of dictionaries
            
        if 'layer_names' in results and results['layer_names']:
            layer_names[opt] = safe_convert(results['layer_names'])
    
    # 1. Training loss comparison
    if train_losses and all(len(losses) > 0 for losses in train_losses.values()):
        try:
            # Find maximum steps across all optimizers
            max_steps = max(len(losses) for losses in train_losses.values())
            x_values = list(range(max_steps))
            
            plot_seaborn_style(
                train_losses,
                x_values,
                "Training Loss Comparison",
                "training_loss_comparison",
                "Loss",
                visuals_dir,
                xlabel="Evaluation Step"
            )
        except Exception as e:
            print(f"Error plotting training losses: {e}")
    else:
        print("Warning: No valid training loss data available")
    
    # 2. Validation loss comparison
    if dev_losses and all(len(losses) > 0 for losses in dev_losses.values()):
        try:
            # Find maximum steps across all optimizers
            max_steps = max(len(losses) for losses in dev_losses.values())
            x_values = list(range(max_steps))
            
            plot_seaborn_style(
                dev_losses,
                x_values,
                "Validation Loss Comparison",
                "validation_loss_comparison",
                "Loss",
                visuals_dir,
                xlabel="Evaluation Step"
            )
        except Exception as e:
            print(f"Error plotting validation losses: {e}")
    else:
        print("Warning: No valid validation loss data available")
    
    # 3. Iteration loss comparison
    if iteration_losses and all(len(losses) > 0 for losses in iteration_losses.values()):
        try:
            # Find minimum number of iterations across optimizers
            min_iters = min(len(losses) for losses in iteration_losses.values())
            
            # Truncate data to minimum length
            truncated_losses = {opt: losses[:min_iters] for opt, losses in iteration_losses.items()}
            x_values = list(range(min_iters))
            
            plot_seaborn_style(
                truncated_losses,
                x_values,
                "Training Loss per Iteration",
                "iteration_loss_comparison",
                "Loss",
                visuals_dir,
                xlabel="Iteration"
            )
            
            # 4. Wall time vs loss comparison
            if wall_times and all(len(times) > 0 for times in wall_times.values()):
                # Normalize wall times to start from 0
                normalized_times = {}
                for opt, times in wall_times.items():
                    if len(times) >= min_iters and times[0] is not None:
                        normalized_times[opt] = [t - times[0] for t in times[:min_iters]]
                
                if normalized_times:
                    plot_seaborn_style(
                        truncated_losses,
                        normalized_times,
                        "Training Loss vs Wall Time",
                        "walltime_loss_comparison",
                        "Loss",
                        visuals_dir,
                        xlabel="Wall Time (seconds)"
                    )
        except Exception as e:
            print(f"Error plotting iteration data: {e}")
    else:
        print("Warning: No valid iteration loss data available")
    
    # 5. Gradient norm visualizations
    if grad_norms and layer_names:
        try:
            visualize_gradient_norms(
                grad_norms,
                layer_names,
                visuals_dir,
                "Gradient Norm Comparison"
            )
        except Exception as e:
            print(f"Error plotting gradient norms: {e}")
    
    # 6. Save metrics to CSV
    try:
        results_dir = os.path.join(os.path.dirname(visuals_dir), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for CSV
        all_metrics = {}
        for opt in all_results:
            if opt in train_losses and opt in dev_losses:
                metrics_list = []
                train_data = train_losses[opt]
                dev_data = dev_losses[opt]
                
                # Get eval steps or generate sequential steps
                eval_steps = safe_convert(all_results[opt].get('eval_steps', range(len(train_data))))
                
                # Ensure all lists are the same length
                min_len = min(len(train_data), len(dev_data), len(eval_steps))
                
                for i in range(min_len):
                    metrics_list.append({
                        'step': eval_steps[i],
                        'train_loss': train_data[i],
                        'dev_loss': dev_data[i]
                    })
                
                all_metrics[opt] = metrics_list
        
        save_experiment_results(all_metrics, results_dir, "optimizer_comparison_metrics.csv")
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")

def main():
    """Main function to train with multiple optimizers and create comparative visualizations."""
    parser = argparse.ArgumentParser(description="Train transformer models with multiple optimizers")
    parser.add_argument(
        "--optimizers", 
        type=str, 
        nargs='+',
        default=config['optimizers_to_train'],
        help="List of optimizers to train and compare"
    )
    args = parser.parse_args()
    
    # Create visualization directory
    visuals_dir = os.path.join(os.path.dirname(config['t_out_path']), config['visuals_dir'])
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Train with each optimizer
    all_results = {}
    for optimizer_name in args.optimizers:
        if optimizer_name not in config['optimizer_params']:
            print(f"Warning: Optimizer {optimizer_name} not found in config. Skipping.")
            continue
        
        # Train the model with the current optimizer
        results = train_with_optimizer(optimizer_name)
        all_results[optimizer_name] = results
    
    # Create comparative visualizations
    if all_results:
        create_comparative_visualizations(all_results, visuals_dir)
        print(f"Comparative visualizations saved to {visuals_dir}")
    else:
        print("No models were trained. Cannot create visualizations.")

if __name__ == "__main__":
    main()
