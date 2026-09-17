"""Test meta-training speed for one episode."""
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split
from trench_ids.cl.device import resolve_device
from trench_ids.model.transfer_adapter import TransferAdapter

device = resolve_device('cpu')
model, _, _ = load_checkpoint(Path('runs/gnn_3layer_residual_s42/checkpoint_task_3.pt'), Path('data/graphs'), 'cpu')
print('Model loaded')

# Create adapter
adapter = TransferAdapter(hidden_dim=64, adapter_dim=32).to('cpu')

# Get fusion module
fusion_module = model.layers[0].fusion['flow']

# Create simple fusion function
def fuse_adapted_simple(adapted):
    embeddings = [adapted[r] for r in ['originates', 'terminated_by', 'targeted_by', 'protocol_of', 'service_of']]
    return torch.stack(embeddings).mean(0)

# Get task signature
task_sig = torch.randn(137)

# Get data
train_graphs = load_split(Path('data/graphs'), 4, 'train')[:20]
test_graphs = load_split(Path('data/graphs'), 4, 'test')[:10]

# Inner loop test
optimizer = torch.optim.SGD(list(adapter.parameters()), lr=0.01)

import time
start = time.time()

# Inner loop (3 steps)
for step in range(3):
    for batch in DataLoader(load_split(Path('data/graphs'), 4, 'train')[:10], batch_size=4, shuffle=True):
        batch = batch.to('cpu')
        with torch.no_grad():
            output = model(batch)
            relation_embeds = output.relations["flow"]
        
        # Forward through adapter
        adapted = {}
        for r, emb in relation_embeds.items():
            if r in ['originates', 'terminated_by', 'targeted_by', 'protocol_of', 'service_of']:
                delta = adapter.relation_adapters[r](emb)
                adapted[r] = emb + 0.5 * delta  # Simple gate = 0.5
        
        # Fuse
        adapted_embeds = [adapted[r] for r in ['originates', 'terminated_by', 'targeted_by', 'protocol_of', 'service_of']]
        fused = torch.stack(adapted_embeds).mean(0)
        
        logits = torch.nn.Linear(64, 11)(fused)
        loss = F.cross_entropy(logits, batch["flow"].y)
        loss.backward()

print(f'3 inner steps done in {time.time() - start:.1f}s')