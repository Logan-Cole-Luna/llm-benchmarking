# --- Configuration ---

# Define vocabulary size and transformer configuration (Small model for testing)
VOCAB_SIZE = 50304          # Number of unique tokens in the vocabulary
CONTEXT_LENGTH = 64         # Maximum sequence length for the model
N_EMBED = 64                # Dimension of the embedding space (reduced from 128)
N_HEAD = 4                  # Number of attention heads in each transformer block (reduced from 8)
N_BLOCKS = 1                # Number of transformer blocks in the model

# Paths to training and development datasets
TRAIN_PATH = "data/train/pile_train.h5"  # File path for the training dataset
DEV_PATH = "data/val/pile_dev.h5"      # File path for the validation dataset

# Transformer training parameters
T_BATCH_SIZE = 16           # Number of samples per training batch (reduced from 32)
T_CONTEXT_LENGTH = 16       # Context length for training batches
T_TRAIN_STEPS = 1000       # Total number of training steps (reduced from 200000)
T_EVAL_STEPS = 200          # Frequency (in steps) to perform evaluation (reduced from 1000)
T_EVAL_ITERS = 50           # Number of iterations to evaluate the model (reduced from 250)
T_LR_DECAY_STEP = 5000      # Step at which to decay the learning rate (reduced from 50000)
T_LR = 5e-4                 # Initial learning rate for training
T_LR_DECAYED = 5e-5         # Learning rate after decay
T_OUT_PATH = "models/transformer_tiny.pt"  # Path to save the trained model

# List of optimizers to train and compare
OPTIMIZERS_TO_TRAIN = ["ADAMW", "ADAM", "SGD", "GGD", "GGD_LW", "GGD_HYBRID", "GGD_ADAM"]

# Optimizer configuration
OPTIMIZER = "GGD_HYBRID"         # Options: "ADAMW", "SGD", "ADAM", "GGD", "GGD_LW", "GGD_HYBRID", "GGD_ADAM"
OPTIMIZER_PARAMS = {
    "ADAMW": {"lr": T_LR, "weight_decay": 0.01},
    "SGD": {"lr": 0.1, "momentum": 0.9, "nesterov": True},
    "ADAM": {"lr": T_LR},
    "GGD": {"normalize": True, "layer_wise": False, "scale_aware": False, 
            "scale_factor": 0.2, "max_group_size": 5000, "adaptive": True, "clip_norm": 1.0,
            "momentum": 0.9, "adaptive_eps": 1e-8, "weight_decay": 0.0005, "lamb": True},
    #"GGD_LW": {"normalize": True, "layer_wise": True, "scale_aware": True, 
    #           "scale_factor": 0.2, "max_group_size": 5000, "adaptive": True, "clip_norm": 1.0, 
    #           "momentum": 0.9, "adaptive_eps": 1e-8, "weight_decay": 0.0005},
    
    "GGD_LW": {
        "lr": 0.1,
        "normalize": True,
        "layer_wise": True,
        "scale_aware": False,
        "max_group_size": 5000,
        "adaptive": False,
        "clip_norm": 0,  #// or use a higher value like 5.0 if clipping is desired
        "momentum": 0.9,
        "weight_decay": 0.0,
        "nesterov": True  #// if your implementation supports it
        },

    "GGD_HYBRID": {"lr": T_LR, "normalize": True, "layer_wise": True, "scale_aware": True, 
                  "scale_factor": 0.2, "use_second_moment": True, "hybrid_mode": True,
                  "second_moment_factor": 0.7, "betas": (0.9, 0.999), "adamw_mode": True,
                  "eps": 1e-8, "weight_decay": 0.01, "clip_norm": 1.0, "lamb": True},
    "GGD_ADAM": {"lr": T_LR, "normalize": True, "layer_wise": True, "scale_aware": True, 
                "scale_factor": 0.2, "use_second_moment": True, "betas": (0.9, 0.999),
                "adamw_mode": True, "eps": 1e-8, "weight_decay": 0.01, "clip_norm": 1.0, "lamb": True}
}

# Device configuration
DEVICE = 'cuda'

# Visualization folder
VISUALS_DIR = "visuals"

# Store all configurations in a dictionary for easy access and modification
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
    'optimizer': OPTIMIZER,
    'optimizer_params': OPTIMIZER_PARAMS,
    'optimizers_to_train': OPTIMIZERS_TO_TRAIN,
    'visuals_dir': VISUALS_DIR
}