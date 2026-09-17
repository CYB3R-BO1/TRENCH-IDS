"""Test baseline evaluation speed."""
from pathlib import Path
import time
import torch

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split
from trench_ids.cl.device import resolve_device
from trench_ids.cl.task_signature import extract_task_signature
from trench_ids.cl.meta_transfer import evaluate_condition
from trench_ids.model.transfer_adapter import fuse_adapted, LightweightClassifierHead

device = resolve_device('cpu')
model, base_classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
model.eval()
base_classifier.eval()

target_support = load_split(Path('data/graphs'), 4, 'train')[:20]
target_query = load_split(Path('data/graphs'), 4, 'test')[:10]

print("Testing evaluate_condition baseline...")
start = time.time()

# We need to import evaluate_condition properly
from trench_ids.cl.meta_transfer import evaluate_condition
from trench_ids.model.transfer_adapter import fuse_adapted, LightweightClassifierHead

result = evaluate_condition(
    model=model,
    base_classifier=base_classifier,
    adapter=None,
    target_support=load_split(Path('data/graphs'), 4, 'train')[:20],
    target_query=load_split(Path('data/graphs'), 4, 'test')[:10],
    task_signature=torch.randn(137),
    device='cpu',
    batch_size=32,
    k_steps=3,
    inner_lr=0.01,
    replay_buffer=[],
    replay_fraction=0.3,
    replay_selection='uniform',
    seed=42,
    out_dir=Path('runs/test_baseline')
)

print('Baseline eval done')