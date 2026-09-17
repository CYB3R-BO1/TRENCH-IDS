"""Profile signature extraction and meta-training on CUDA."""
import sys
sys.path.insert(0, 'src')
import torch
import time
from pathlib import Path

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.device import resolve_device
from trench_ids.cl.episode_generator import EpisodeGenerator
from trench_ids.cl.task_signature import extract_task_signature

device = resolve_device('cuda')
print('Device:', device)

# Load checkpoint
model, classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cuda')
model.eval()
print('Model loaded')

# Load episodes
generator = EpisodeGenerator(Path('data/graphs'))
generator.initialize()
meta_train, meta_val, _ = generator.generate_episodes(n_train=5, n_val=1)

print(f'Loaded {len(meta_train)} meta-train episodes')

# Time signature extraction
import time
start = time.time()
for i, ep in enumerate(meta_train[:2]):
    start = time.time()
    sig = extract_task_signature(ep['target_support'], model, 'cuda', stats=None, include_covariance=False, batch_size=32)
    print(f'Episode {i+1} signature extraction: {time.time() - start:.1f}s')
print(f'Total signature extraction: {time.time() - start:.1f}s')