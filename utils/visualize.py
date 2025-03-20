import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union

# Add a utility function at the top of the file, after imports
def ensure_list(data):
    """Convert torch tensors or numpy arrays to Python lists."""
    if isinstance(data, torch.Tensor):
        return data.cpu().tolist() if data.numel() > 0 else []
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return data
    else:
        return [data]  # Convert scalar to list

# Set up seaborn style for consistent, professional-looking visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def setup_visualization_dir(base_dir: str) -> str:
    """Create and return a directory for saving visualizations."""
    visuals_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(visuals_dir, exist_ok=True)
    return visuals_dir

def _ensure_native_types(data):
    """
    Ensure all data is converted to Python native types, handling tensors properly.
    
    Args:
        data: Any data structure that might contain tensors
        
    Returns:
        Data with all tensors converted to Python native types
    """
    if isinstance(data, torch.Tensor):
        # Handle 0-dimensional tensors (scalars)
        if data.dim() == 0:
            return data.item()
        # Handle multi-dimensional tensors
        return data.detach().cpu().tolist()
    elif isinstance(data, (list, tuple)):
        return [_ensure_native_types(item) for item in data]
    elif isinstance(data, dict):
        return {k: _ensure_native_types(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# Update the plot_seaborn_style function to ensure data is properly converted
def plot_seaborn_style(
    y_data: Dict[str, List[float]],
    x_data: Union[List[int], Dict[str, List[float]]],
    title: str,
    filename: str,
    y_label: str,
    save_dir: str,
    xlabel: str = "Steps"
) -> None:
    """
    Plot data in Seaborn style and save to file.
    
    Args:
        y_data: Dictionary mapping optimizer names to y-values
        x_data: List of x-values or dictionary mapping optimizer names to x-values
        title: Plot title
        filename: Output filename
        y_label: Label for y-axis
        save_dir: Directory to save plots
        xlabel: Label for x-axis
    """
    plt.figure(figsize=(12, 6))
    
    # Convert all data to native Python types to prevent tensor issues
    y_data = _ensure_native_types(y_data)
    x_data = _ensure_native_types(x_data)
    
    # Process the data into a format suitable for Seaborn
    all_x = []
    all_y = []
    all_optimizers = []
    
    for optimizer_name, y_values in y_data.items():
        if not y_values:  # Skip if no data
            continue
            
        if isinstance(x_data, dict):
            # If x_data is a dictionary, use the values for this optimizer
            x_values = x_data.get(optimizer_name, list(range(len(y_values))))
            # Ensure x_values is the right length
            x_values = x_values[:len(y_values)]
        else:
            # Otherwise use the same x values for all optimizers
            # Ensure x_data is the right length
            x_values = x_data[:len(y_values)] if len(x_data) > len(y_values) else x_data
            
            # If x_data is too short, extend it
            if len(x_values) < len(y_values):
                x_values = list(range(len(y_values)))
        
        all_x.extend(x_values)
        all_y.extend(y_values)
        all_optimizers.extend([optimizer_name] * len(y_values))
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        'x': all_x,
        'y': all_y,
        'optimizer': all_optimizers
    })
    
    # Create plot
    sns.lineplot(
        data=df,
        x='x',
        y='y',
        hue='optimizer',
        style='optimizer',
        markers=True,
        dashes=False,
        palette='tab10'
    )
    
    # Set plot labels and title
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Optimizer', fontsize=12)
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_gradient_norms(
    grad_norms_history: Dict[str, List[Dict[str, float]]],
    layer_names_dict: Dict[str, List[str]],
    visuals_dir: str,
    experiment_title: str = "Transformer Training"
) -> None:
    """
    Create visualizations for gradient norms per layer and optimizer.
    
    Args:
        grad_norms_history: Dictionary mapping optimizer names to lists of dictionaries mapping layer names to gradient norms
        layer_names_dict: Dictionary mapping optimizer names to lists of layer names
        visuals_dir: Directory to save visualizations
        experiment_title: Title for the plots
    """
    # Create a subdirectory for gradient norm visualizations
    grad_dir = os.path.join(visuals_dir, "gradient_norms")
    os.makedirs(grad_dir, exist_ok=True)
    
    # Find common layers across all optimizers
    all_layers = set()
    for opt in layer_names_dict:
        all_layers.update(layer_names_dict[opt])
    all_layers = sorted(list(all_layers))
    
    # 1. Boxplot comparison of gradient norms per layer per optimizer
    boxplot_data = []
    for opt in grad_norms_history:
        for layer in all_layers:
            if layer in layer_names_dict[opt]:
                # Extract all gradient norms for this layer and optimizer
                norms = [epoch_norms[layer] for epoch_norms in grad_norms_history[opt] 
                         if layer in epoch_norms]
                if norms:
                    # Add data points for the boxplot
                    for norm in norms:
                        boxplot_data.append({
                            'Optimizer': opt,
                            'Layer': layer, 
                            'Gradient Norm': norm
                        })
    
    # Create dataframe for seaborn
    df_norms = pd.DataFrame(boxplot_data)
    
    # Only create plot if we have data
    if not df_norms.empty:
        # Use seaborn's boxplot with consistent styling
        plt.figure(figsize=(14, 8), dpi=300)
        ax = plt.gca()
        sns.boxplot(x='Layer', y='Gradient Norm', hue='Optimizer', data=df_norms, ax=ax)
        
        plt.title(f'{experiment_title}: Gradient Norm Distribution per Layer', fontweight='bold', fontsize=13)
        plt.xlabel('Layer', fontweight='bold')
        plt.ylabel('Gradient Norm', fontweight='bold')
        plt.xticks(rotation=45)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
        legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(grad_dir, 'gradient_norm_boxplot.png'), bbox_inches='tight')
        plt.savefig(os.path.join(grad_dir, 'gradient_norm_boxplot.pdf'), bbox_inches='tight')
        plt.close()
    
    # 2. Line plots showing gradient norm evolution over iterations for each layer
    if all_layers:
        n_layers = len(all_layers)
        n_cols = min(3, n_layers)  # Maximum of 3 columns
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, n_rows * 4), dpi=300)
        for i, layer in enumerate(all_layers):
            plt.subplot(n_rows, n_cols, i + 1)
            ax = plt.gca()
            palette = sns.color_palette("colorblind")
            
            for j, opt in enumerate(grad_norms_history):
                if layer in layer_names_dict[opt]:
                    # Extract gradient norms for this layer and optimizer
                    norms = [epoch_norms.get(layer, 0) for epoch_norms in grad_norms_history[opt]]
                    # Plot moving average for smoother visualization
                    window_size = min(20, len(norms))
                    if window_size > 0:
                        norms_smoothed = pd.Series(norms).rolling(window=window_size, min_periods=1).mean().values
                        sns.lineplot(x=range(len(norms_smoothed)), y=norms_smoothed, 
                                    label=opt, color=palette[j % 10], linewidth=2, alpha=0.9)
            
            plt.title(f'Layer: {layer}', fontweight='bold')
            plt.xlabel('Iteration', fontweight='bold')
            plt.ylabel('Gradient Norm', fontweight='bold')
            plt.yscale('log')  # Log scale for better visualization
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if i == 0:  # Only add legend to first subplot to avoid redundancy
                legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, 
                                   framealpha=0.95, facecolor='white')
                legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(grad_dir, 'gradient_norm_evolution.png'), bbox_inches='tight')
        plt.savefig(os.path.join(grad_dir, 'gradient_norm_evolution.pdf'), bbox_inches='tight')
        plt.close()
    
    # 3. Ratio of max/min gradient norms to show imbalance
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    palette = sns.color_palette("colorblind")
    
    for i, opt in enumerate(grad_norms_history.keys()):
        # For each iteration, compute max/min ratio across layers
        ratios = []
        for epoch_norms in grad_norms_history[opt]:
            norms = [epoch_norms[layer] for layer in epoch_norms if epoch_norms[layer] > 0]
            if len(norms) > 1:  # Need at least 2 layers to compute ratio
                max_norm = max(norms)
                min_norm = min(norms)
                if min_norm > 0:  # Avoid division by zero
                    ratios.append(max_norm / min_norm)
        
        # Plot moving average of ratios
        if ratios:
            window_size = min(20, len(ratios))
            ratios_smoothed = pd.Series(ratios).rolling(window=window_size, min_periods=1).mean().values
            sns.lineplot(x=range(len(ratios_smoothed)), y=ratios_smoothed, 
                        label=opt, color=palette[i % 10], linewidth=2, alpha=0.9)
    
    plt.title(f'{experiment_title}: Max/Min Gradient Norm Ratio (Layer Imbalance)', fontweight='bold', fontsize=13)
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Max/Min Gradient Norm Ratio', fontweight='bold')
    plt.yscale('log')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    legend = plt.legend(title='Optimizer', frameon=True, fancybox=True, framealpha=0.95, facecolor='white')
    legend.get_title().set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(grad_dir, 'gradient_imbalance_ratio.png'), bbox_inches='tight')
    plt.savefig(os.path.join(grad_dir, 'gradient_imbalance_ratio.pdf'), bbox_inches='tight')
    plt.close()

def save_experiment_results(data: Dict[str, List[Dict[str, Any]]], results_dir: str, csv_filename: str) -> None:
    """
    Save experiment results to a CSV file.
    
    Args:
        data: Dictionary mapping optimizer names to lists of dictionaries containing metrics
        results_dir: Directory to save the CSV file
        csv_filename: Name of the CSV file
    """
    # Flatten the data for CSV
    flat_data = []
    for optimizer, metrics_list in data.items():
        for step, metrics in enumerate(metrics_list):
            flat_data.append({
                'optimizer': optimizer,
                'step': step,
                **metrics
            })
    
    # Create dataframe and save to CSV
    df = pd.DataFrame(flat_data)
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, csv_filename), index=False)

class GradientTracker:
    """
    Class to track and record gradient norms during training.
    """
    def __init__(self):
        self.gradient_norms = []
        self.layer_names = set()
        
    def update(self, model: torch.nn.Module) -> None:
        """
        Record gradient norms for the current step.
        
        Args:
            model: PyTorch model with parameters that have gradients
        """
        current_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                current_norms[name] = norm
                self.layer_names.add(name)
        
        self.gradient_norms.append(current_norms)
    
    def get_data(self) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Return the collected gradient norm data.
        
        Returns:
            Tuple containing the gradient norms history and the list of layer names
        """
        return self.gradient_norms, list(self.layer_names)

# Update compare_optimizers_from_checkpoints function to ensure data is properly converted
def compare_optimizers_from_checkpoints(checkpoints_dir: str, output_dir: str, prefix: str = "transformer_tiny") -> None:
    """
    Load checkpoints for different optimizers and create comparative visualizations.
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        output_dir: Directory to save visualizations
        prefix: Prefix of checkpoint files
    """
    import glob
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all optimizer checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, f"{prefix}_*.pt"))
    optimizers = []
    
    for file in checkpoint_files:
        # Extract optimizer name from filename
        filename = os.path.basename(file)
        if "_best_" in filename:
            continue  # Skip best checkpoints for now
        
        parts = filename.replace('.pt', '').split('_')
        if len(parts) >= 2:
            optimizer = parts[-1]
            optimizers.append(optimizer)
    
    if not optimizers:
        print(f"No optimizer checkpoints found in {checkpoints_dir} with prefix {prefix}")
        return
    
    # Load metrics from each optimizer
    train_losses = {}
    dev_losses = {}
    eval_steps = {}
    iterations = {}
    walltimes = {}
    grad_norms = {}
    layer_names = {}
    
    for optimizer in optimizers:
        checkpoint_path = os.path.join(checkpoints_dir, f"{prefix}_{optimizer}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Extract metrics if available
            if 'training_metrics' in checkpoint:
                metrics = checkpoint['training_metrics']
                train_losses[optimizer] = ensure_list(metrics.get('eval_train_losses', []))
                dev_losses[optimizer] = ensure_list(metrics.get('eval_dev_losses', []))
                eval_steps[optimizer] = ensure_list(metrics.get('eval_steps', []))
                iterations[optimizer] = ensure_list(metrics.get('iteration_losses', []))
                walltimes[optimizer] = ensure_list(metrics.get('wall_times', []))
                grad_norms[optimizer] = ensure_list(metrics.get('gradient_norms', []))
                layer_names[optimizer] = ensure_list(metrics.get('layer_names', []))
                
                print(f"Loaded metrics for optimizer {optimizer}")
            else:
                print(f"No training metrics found in checkpoint for {optimizer}")
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
    
    # Create comparative visualizations
    if train_losses:
        # 1. Training loss comparison
        plot_seaborn_style(
            train_losses,
            range(max([len(losses) for losses in train_losses.values()])),
            "Training Loss Comparison",
            "training_loss_comparison",
            "Loss",
            output_dir,
            xlabel="Evaluation Step"
        )
        
        # 2. Validation loss comparison
        plot_seaborn_style(
            dev_losses,
            range(max([len(losses) for losses in dev_losses.values()])),
            "Validation Loss Comparison",
            "validation_loss_comparison",
            "Loss",
            output_dir,
            xlabel="Evaluation Step"
        )
        
        # 3. Raw iteration loss comparison (limit to avoid excessive data)
        if iterations:
            min_iterations = min([len(its) for its in iterations.values()])
            limited_iterations = {opt: its[:min_iterations] for opt, its in iterations.items()}
            
            plot_seaborn_style(
                limited_iterations,
                range(min_iterations),
                "Training Loss per Iteration",
                "iteration_loss_comparison",
                "Loss",
                output_dir,
                xlabel="Iteration"
            )
        
        # 4. Wall time comparison
        if walltimes:
            # Normalize wall times to start from 0 for each optimizer
            normalized_walltimes = {}
            for opt, times in walltimes.items():
                if times:
                    start_time = times[0]
                    normalized_walltimes[opt] = [t - start_time for t in times[:min_iterations]]
            
            if normalized_walltimes:
                wall_data = {opt: iterations[opt][:min_iterations] for opt in iterations}
                
                plot_seaborn_style(
                    wall_data,
                    normalized_walltimes,
                    "Training Loss vs Wall Time",
                    "walltime_loss_comparison",
                    "Loss",
                    output_dir,
                    xlabel="Wall Time (seconds)"
                )
        
        # 5. Gradient norm visualizations
        if grad_norms:
            visualize_gradient_norms(
                grad_norms,
                layer_names,
                output_dir,
                "Gradient Norm Comparison"
            )
        
        print(f"Comparative visualizations saved to {output_dir}")
    else:
        print("No metrics found in checkpoints. Cannot create visualizations.")
