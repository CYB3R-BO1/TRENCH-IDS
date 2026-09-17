"""Test meta-training speed."""
from pathlib import Path
import time
import torch

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split
from trench_ids.cl.device import resolve_device
from trench_ids.cl.task_signature import extract_task_signature, SignatureStats
from trench_ids.cl.meta_transfer import normalize_signature
from trench_ids.cl.episode_generator import EpisodeGenerator
from trench_ids.cl.fomaml import run_fomaml_episode
from trench_ids.model.transfer_adapter import TransferAdapter, LightweightClassifierHead, fuse_adapted

device = resolve_device('cpu')
model, base_classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
model.eval()
base_classifier.eval()

# Create adapter and classifier head
adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to('cpu')
classifier_head = LightweightClassifierHead(input_dim=64, num_classes=11).to('cpu')

# Load meta-episodes
from trench_ids.cl.episode_generator import EpisodeGenerator
generator = EpisodeGenerator(Path('data/graphs'))
generator.initialize(excluded_flow_ids=set())
meta_train, meta_val, _ = generator.generate_episodes(n_train=5, n_val=1)

print(f'Loaded {len(meta_train)} meta-train episodes')

# Extract signatures
signatures = []
for ep in meta_train:
    sig = extract_task_signature(ep['target_support'], model, 'cpu', stats=None, include_covariance=False, batch_size=32)
    signatures.append(sig)
print('Signatures extracted')

# Test one meta-training step
from trench_ids.cl.fomaml import run_fomaml_episode
import time

start = time.time()
for i, episode in enumerate(meta_train[:2]):
    print(f'Episode {i+1}/2...')
    start = time.time()
    meta_loss, meta_grads = run_fomaml_episode(
        model=model,
        adapter=adapter,
        classifier_head=classifier_head,
        source_checkpoint=str(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt')),
        target_support=episode['target_support'],
        target_query=episode['target_query'],
        task_signature=torch.randn(137),
        k_steps=3,
        alpha=0.01,
        batch_size=32,
        device='cpu'
    )
    print(f'  Episode {i+1}: loss={meta_loss:.4f}, time={time.time()-start:.1f}s')

print('Done')