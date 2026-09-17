import sys
sys.path.insert(0, 'src')
import json
import torch
from pathlib import Path
from trench_ids.vocab import load_vocab
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.model.flat import FlatFlowEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ' + str(device))
if torch.cuda.is_available():
    print('gpu: ' + torch.cuda.get_device_name(0))

graphs = torch.load(Path('data/graphs/task_3_train.pt'), weights_only=False)
g = graphs[0]
print('sample graph task3: flows=%d classes=%s' % (g['flow'].y.numel(), sorted(set(g['flow'].y.tolist()))))
vocab = load_vocab(Path('data/graphs/vocab.json'))
model = RelationSpecificHeteroGNN.from_graph(g, hidden_dim=64, protocol_vocab_size=len(vocab['PROTOCOL']), service_vocab_size=len(vocab['L7_PROTO']), num_layers=3, attn_dim=128, port_tail_buckets=32, fusion='attention', use_residual=True).to(device)
model.eval()
with torch.no_grad():
    out = model(g.to(device))
print('fused flow shape: ' + str(tuple(out.fused['flow'].shape)))
print('relations flow keys: ' + str(sorted(out.relations['flow'].keys())))
n_params = sum(p.numel() for p in model.parameters())
print('n_params_3layer: %d' % n_params)
# Flat baseline
flat = FlatFlowEncoder(hidden_dim=64, protocol_vocab_size=len(vocab['PROTOCOL']), service_vocab_size=len(vocab['L7_PROTO']), use_host_features=True, mlp_hidden=254).to(device)
flat.eval()
with torch.no_grad():
    fout = flat(g.to(device))
print('flat fused flow shape: ' + str(tuple(fout.fused['flow'].shape)))
with open('validation_task_pairs_2026-09-09/07_model_gpu.json', 'w') as f:
    json.dump({'device': str(device), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'fused_shape': list(out.fused['flow'].shape), 'relations': sorted(out.relations['flow'].keys()), 'n_params_3layer': n_params, 'flat_shape': list(fout.fused['flow'].shape)}, f, indent=2)
