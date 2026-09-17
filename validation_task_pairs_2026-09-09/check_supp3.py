import json
from pathlib import Path
import torch
from collections import defaultdict

# 1) incremental forgetting per prefix from forgetting_matrix.json
fm = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(open('runs/gnn_3layer_residual_s42/forgetting_matrix.json')).items()}
def avg_forget(sub):
    T = max(sub.keys())
    if T == 1:
        return 0.0
    vals = []
    for i in range(1, T):
        peak = max(sub[t][i] for t in range(i, T))
        vals.append(peak - sub[T][i])
    return sum(vals) / len(vals)
print('INCREMENTAL (aggregate Eq1 per prefix):')
for T in range(1, 7):
    sub = {t: {i: fm[t][i] for i in fm[t] if i <= T} for t in range(1, T + 1)}
    print('after T%d: %.4f' % (T, avg_forget(sub)))
print('per-task drops T6: ', end='')
T = 6
for i in range(1, 7):
    peak = max(fm[t][i] for t in range(i, 6))
    print('T%d %.4f (peak %.4f final %.4f)' % (i, peak - fm[6][i], peak, fm[6][i]), end=' | ')
print()

# 2) param breakdown with current vocab
import sys
sys.path.insert(0, 'src')
from trench_ids.vocab import load_vocab
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
vocab = load_vocab(Path('data/graphs/vocab.json'))
graphs = torch.load(Path('data/graphs/task_1_train.pt'), weights_only=False)
g = graphs[0]
for nl in (1, 2, 3):
    m = RelationSpecificHeteroGNN.from_graph(g, hidden_dim=64, protocol_vocab_size=len(vocab['PROTOCOL']), service_vocab_size=len(vocab['L7_PROTO']), num_layers=nl, attn_dim=128, port_tail_buckets=32, fusion='attention', use_residual=(nl > 1))
    enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith('encoders.'))
    conv = sum(p.numel() for n, p in m.named_parameters() if '.conv.' in n)
    fus = sum(p.numel() for n, p in m.named_parameters() if '.fusion.' in n)
    tot = sum(p.numel() for p in m.parameters())
    print('layers=%d total=%d enc=%d conv=%d fus=%d' % (nl, tot, enc, conv, fus))
