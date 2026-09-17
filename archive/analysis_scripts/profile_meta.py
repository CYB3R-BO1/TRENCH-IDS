"""Profile meta-training to find bottlenecks."""
from pathlib import Path
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split
from trench_ids.cl.device import resolve_device
from trench_ids.cl.task_signature import extract_task_signature, SignatureStats
from trench_ids.cl.meta_transfer import normalize_signature
from trench_ids.cl.episode_generator import EpisodeGenerator
from trench_ids.cl.fomaml import run_fomaml_episode
from trench_ids.model.transfer_adapter import TransferAdapter, LightweightClassifierHead, fuse_adapted

device = resolve_device('cpu')
print(f"Device: {device}")

checkpoint_path = Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt')
model, base_classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
model.eval()
base_classifier.eval()

# Create adapter
adapter = TransferAdapter(hidden_dim=64, adapter_dim=32)
classifier_head = torch.nn.Linear(64, 11)

# Load meta-episodes
generator = EpisodeGenerator(Path('data/graphs'))
generator.initialize(excluded_flow_ids=set())
meta_train, meta_val, _ = generator.generate_episodes(n_train=2, n_val=1)

print(f'Loaded {len(meta_train)} meta-train episodes')

# Time signature extraction
start = time.time()
for i, ep in enumerate(meta_train):
    sig = extract_task_signature(ep['target_support'], model, 'cpu', stats=None, include_covariance=False, batch_size=32)
    print(f'Episode {i+1} signature extraction: {time.time() - start:.1f}s')
print(f'Total signature extraction: {time.time() - start:.1f}s')

# Test one FOMAML episode
print('\nTesting FOMAML episode...')
ep = meta_train[0]
start = time.time()

adapter = TransferAdapter(hidden_dim=64, adapter_dim=32)
classifier_head = torch.nn.Linear(64, 11)

meta_loss, meta_grads = run_fomaml_episode(
    model=model,
    adapter=adapter,
    classifier_head=classifier_head,
    source_checkpoint=str(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt')),
    target_support=ep['target_support'],
    target_query=ep['target_query'],
    task_signature=torch.randn(137),
    k_steps=3,
    alpha=0.01,
    batch_size=32,
    device='cpu'
)
print(f'FOMAML episode time: {time.time() - start:.1f}s')
print(f'Meta loss: {meta_loss:.4f}')