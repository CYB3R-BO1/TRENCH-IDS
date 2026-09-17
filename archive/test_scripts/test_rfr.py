import sys
import os

# Set up paths
os.chdir(r"C:\Users\cherr\Documents\PROJECTS\TRENCH-IDS")
sys.path.insert(0, "src")

# Set up command line args for hydra
sys.argv = [
    'train.py',
    'train',
    'rfr.enabled=true',
    'rfr.lambda=0.1',
    'rfr.apply_only_base=true',
    'replay.enabled=true',
    'train.epochs_per_task=1',
    'train.warmup_epochs=1',
    'train.batch_size=32',
    'paths.out_dir=runs/rfr_test'
]

# Change to the correct working directory for hydra config
os.chdir(r"C:\Users\cherr\Documents\PROJECTS\TRENCH-IDS\src\trench_ids\cl")

# Run training
from train import main
main()