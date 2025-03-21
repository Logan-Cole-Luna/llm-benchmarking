import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import time
from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator
from typing import Dict, List, Tuple
from GGD import GGD
from utils.visualize import setup_visualization_dir, plot_seaborn_style, visualize_gradient_norms, save_experiment_results, GradientTracker

# --- Initialize the Model and Print Parameters ---

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
).to(config['device'])

# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")

# --- Training metrics tracking ---
metrics_data = {
    'train_losses': [],
    'dev_losses': [],
    'eval_steps': [],
    'wall_times': [],
    'iteration_losses': []
}

# Initialize gradient tracker
gradient_tracker = GradientTracker()

# --- Optimizer Setup and Loss Tracking ---

optimizer_name = config['optimizer']
optimizer_params = config['optimizer_params'][optimizer_name]

print(f"Using optimizer: {optimizer_name} with parameters: {optimizer_params}")

if optimizer_name == "GGD":
    optimizer = GGD(
            model.parameters(),
            **optimizer_params
        )
elif optimizer_name == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
elif optimizer_name == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
elif optimizer_name == "ADAMW":
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
elif optimizer_name == "GGD_LW":
    optimizer = GGD(
        model.parameters(),
        **optimizer_params
    )
elif optimizer_name == "GGD_HYBRID":
    optimizer = GGD(
        model.parameters(),
        **optimizer_params
    )
elif optimizer_name == "GGD_ADAM":
    optimizer = GGD(
        model.parameters(),
        **optimizer_params
    )

# List to track loss values during training.
losses = []

# Define a window size for averaging recent losses in the training loop.
AVG_WINDOW = 64

# Helper function to estimate the average loss for training and development data.
@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.

    for split in ['train', 'dev']:
        # Select the appropriate data path for the current split.
        data_path = config['train_path'] if split == 'train' else config['dev_path']

        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )

        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break

        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()

    model.train()  # Restore the model to training mode.
    return out

# --- Training Loop ---

# Create a batch iterator for the training data.
batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)

# Create directories for results and visualizations
results_dir = os.path.join(os.path.dirname(config['t_out_path']), "results")
os.makedirs(results_dir, exist_ok=True)
visuals_dir = setup_visualization_dir(os.path.dirname(config['t_out_path']))

# Track start time for walltime tracking
start_time = time.time()

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data.
        xb, yb = next(batch_iterator)

        # Perform a forward pass and compute the loss.
        _, loss = model(xb, yb)

        # Record the loss for tracking.
        losses.append(loss.item())
        metrics_data['iteration_losses'].append(loss.item())
        metrics_data['wall_times'].append(time.time() - start_time)
        
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate the loss and update the model parameters.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Track gradients
        gradient_tracker.update(model)
        
        optimizer.step()

        # Periodically evaluate the model on training and development data.
        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            print(f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
            
            # Record evaluation metrics
            metrics_data['train_losses'].append(train_loss)
            metrics_data['dev_losses'].append(dev_loss)
            metrics_data['eval_steps'].append(step)

        # Decay the learning rate at the specified step.
        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
    except StopIteration:
        # Handle the case where the training data iterator ends early.
        print("Training data iterator finished early.")
        break

# --- Save Model and Final Evaluation ---

# Create the output directory if it does not exist.
os.makedirs(config['t_out_path'].split('/')[0], exist_ok=True)

# Perform a final evaluation of the model on training and development datasets.
evaluation_losses = estimate_loss(200)
train_loss = evaluation_losses['train']
dev_loss = evaluation_losses['dev']

# Ensure unique model save path in case the file already exists.
modified_model_out_path = config['t_out_path']
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"

# Save the model's state dictionary, optimizer state, and training metadata.
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'steps': len(losses),
    },
    modified_model_out_path
)

# Get gradient norm data
gradient_norms, layer_names = gradient_tracker.get_data()

# Create visualizations
if metrics_data['eval_steps']:
    # 1. Training and validation loss curves
    optimizers_data = {config['optimizer']: metrics_data['train_losses']}
    plot_seaborn_style(
        optimizers_data,
        metrics_data['eval_steps'],
        "Training Loss vs Step",
        f"training_loss_{config['optimizer']}",
        "Loss",
        visuals_dir,
        xlabel="Step"
    )
    
    optimizers_data = {config['optimizer']: metrics_data['dev_losses']}
    plot_seaborn_style(
        optimizers_data,
        metrics_data['eval_steps'],
        "Validation Loss vs Step",
        f"validation_loss_{config['optimizer']}",
        "Loss",
        visuals_dir,
        xlabel="Step"
    )
    
    # 2. Raw iteration loss curve
    optimizers_data = {config['optimizer']: metrics_data['iteration_losses']}
    plot_seaborn_style(
        optimizers_data,
        range(len(metrics_data['iteration_losses'])),
        "Training Loss per Iteration",
        f"iteration_loss_{config['optimizer']}",
        "Loss",
        visuals_dir,
        xlabel="Iteration"
    )
    
    # 3. Wall time vs loss curve
    optimizers_data = {config['optimizer']: metrics_data['iteration_losses']}
    wall_times_data = {config['optimizer']: metrics_data['wall_times']}
    plot_seaborn_style(
        optimizers_data,
        wall_times_data,
        "Training Loss vs Wall Time",
        f"walltime_loss_{config['optimizer']}",
        "Loss",
        visuals_dir,
        xlabel="Wall Time (seconds)"
    )
    
    # 4. Gradient norm visualizations
    grad_norms_data = {config['optimizer']: gradient_norms}
    layer_names_data = {config['optimizer']: layer_names}
    visualize_gradient_norms(
        grad_norms_data,
        layer_names_data,
        visuals_dir,
        f"Transformer Training ({config['optimizer']})"
    )
    
    # Save metrics data to CSV
    all_metrics = [{
        'step': step,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'optimizer': config['optimizer']
    } for step, train_loss, dev_loss in zip(
        metrics_data['eval_steps'], 
        metrics_data['train_losses'], 
        metrics_data['dev_losses']
    )]
    save_experiment_results(
        {config['optimizer']: all_metrics},
        results_dir,
        f"metrics_{config['optimizer']}.csv"
    )

print(f"Saved model to {modified_model_out_path}")
print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
print(f"Visualizations saved to {visuals_dir}")