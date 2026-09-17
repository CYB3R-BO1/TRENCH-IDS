"""Test baseline training speed."""
from pathlib import Path
import time
import torch
from torch_geometric.loader import DataLoader

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split
from trench_ids.cl.device import resolve_device

device = resolve_device('cpu')
model, _, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
print('Model loaded')

graphs = load_split(Path('data/graphs'), 4, 'train')[:20]
loader = DataLoader(graphs, batch_size=16, shuffle=True)

head = torch.nn.Linear(64, 11)
optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

import time
start = time.time()
for epoch in range(3):
    for batch in DataLoader(load_split(Path('data/graphs'), 4, 'train')[:20], batch_size=16, shuffle=True):
        batch = batch.to('cpu')
        with torch.no_grad():
            output = model(batch)
        logits = torch.nn.Linear(64, 11)(output.fused['flow'])
        loss = torch.nn.functional.cross_entropy(logits, batch['flow'].y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} done')

print(f'Time: {time.time() - start:.1f}s')