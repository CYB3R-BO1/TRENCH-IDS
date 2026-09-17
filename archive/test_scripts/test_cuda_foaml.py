"""Profile meta-training step on CUDA."""
import sys
sys.path.insert(0, 'src')
import torch
import time
from pathlib import Path

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.device import resolve_device
from trench_ids.cl.episode_generator import EpisodeGenerator
from trench_ids.cl.task_signature import extract_task_signature
from trench_ids.cl.fomaml import run_fomaml_episode
from trench_ids.model.transfer_adapter import TransferAdapter, LightweightClassifierHead

device = resolve_device('cuda')
print('Device:', device)

# Load checkpoint
model, classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cuda')
model.eval()
print('Model loaded')

# Create adapter and classifier head
adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to('cuda')
classifier_head = torch.nn.Linear(64, 11).to('cuda')

# Load episodes
generator = EpisodeGenerator(Path('data/graphs'))
generator.initialize()
meta_train, meta_val, _ = generator.generate_episodes(n_train=2, n_val=1)

print(f'Loaded {len(meta_train)} meta-train episodes')

# Extract signatures
start = time.time()
signatures = []
for ep in meta_train:
    sig = extract_task_signature(ep['target_support'], model, 'cuda', stats=None, include_covariance=False, batch_size=32)
    signatures.append(sig)
print(f'Signatures extracted in {time.time() - start:.1f}s')

# Test one meta-training step
print('\nTesting FOMAML episode...')
ep = meta_train[0]
start = time.time()

# Get task signature
task_sig = torch.randn(137, device='cuda')

# Create adapter and classifier head
adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to('cuda')
classifier_head = torch.nn.Linear(64, 11).to('cuda')

# Test one FOMAML episode
from trench_ids.cl.fomaml import run_fomaml_episode

start = time.time()
meta_loss, meta_grads = run_fomaml_episode(
    model=None,  # Will be loaded inside
    adapter=TransferAdapter(hidden_dim=64, adapter_dim=32).to('cuda'),
    classifier_head=torch.nn.Linear(64, 11).to('cuda'),
    source_checkpoint=str(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt')),
    target_support=ep['target_support'],
    target_query=ep['target_query'],
    task_signature=torch.randn(137, device='cuda'),
    k_steps=3,
    alpha=0.01,
    batch_size=32,
    device='cuda'
)
print(f'FOMAML episode time: {time.time() - start:.1f}s')
print(f'Meta loss: {meta_loss:.4f}')