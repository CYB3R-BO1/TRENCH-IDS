"""Test training speed for TCTRL."""
from pathlib import Path
import time
import torch
from torch_geometric.loader import DataLoader

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, build_replay_augmented_set
from trench_ids.cl.device import resolve_device

device = resolve_device('cpu')
checkpoint_path = Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt')
model, classifier, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
print('Model loaded')

target_support = load_split(Path('data/graphs'), 4, 'train')
target_query = load_split(Path('data/graphs'), 4, 'test')

replay_buffer_path = Path('runs/gnn_3layer_residual_s42/replay_buffer_task_3.pt')
if replay_buffer_path.exists():
    replay_buffer = torch.load(replay_buffer_path, weights_only=False)
else:
    replay_buffer = []

training_set = build_replay_augmented_set(load_split(Path('data/graphs'), 4, 'train'), [], 0.3)
print(f'Training set size: {len(training_set)}')

from torch_geometric.loader import DataLoader
loader = DataLoader(load_split(Path('data/graphs'), 4, 'train')[:100], batch_size=32, shuffle=True)
query_loader = DataLoader(load_split(Path('data/graphs'), 4, 'test'), batch_size=32, shuffle=False)

head = torch.nn.Linear(64, 11)
optimizer = torch.optim.SGD(list(head.parameters()), lr=0.01)

import time
start = time.time()
for epoch in range(3):
    for batch in DataLoader(load_split(Path('data/graphs'), 4, 'train')[:100], batch_size=32, shuffle=True):
        batch = batch.to('cpu')
        optimizer.zero_grad()
        with torch.no_grad():
            output = model(batch)
        logits = torch.nn.Linear(64, 11)(output.fused['flow'])
        loss = torch.nn.functional.cross_entropy(logits, batch['flow'].y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} done')
print(f'Time: {time.time() - start:.1f}s')