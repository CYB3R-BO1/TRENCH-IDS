import json
from pathlib import Path
import torch

gc = json.load(open('data/graphs/graph_counts.json'))
print('graph_size=%s benign_ratio=%s' % (str(gc.get('graph_size')), str(gc.get('benign_ratio'))))
print('downsampling=%s' % str(gc.get('downsampling')))

tot_g, tot_f, tot_a, tot_b = 0, 0, 0, 0
for t in range(1, 7):
    tg = sum(gc[str(t)][s]['num_graphs'] for s in ('train', 'val', 'test'))
    cc = {}
    for s in ('train', 'val', 'test'):
        r = gc[str(t)][s]
        print('T%d %s: graphs=%d flows/graph mean=%.1f ratio=%s class_counts=%s' % (
            t, s, r['num_graphs'], r['flows_per_graph']['mean'],
            str(r.get('realised_attack_benign_ratio')), str(r.get('class_counts'))))
        for k, v in r.get('class_counts', {}).items():
            cc[k] = cc.get(k, 0) + v
    tot_g += tg
    a = sum(v for k, v in cc.items() if k != 'Benign')
    b = cc.get('Benign', 0)
    tot_a += a
    tot_b += b
    tot_f += a + b
    print('  T%d TOTAL: graphs=%d flows=%d attack=%d benign=%d ratio=%.4f' % (t, tg, a + b, a, b, (a / b if b else float('inf'))))
print('GRAND: graphs=%d flows=%d attack=%d benign=%d ratio=%.4f' % (tot_g, tot_f, tot_a, tot_b, tot_a / tot_b))

# cross-check .pt files on disk
import glob
files = sorted(glob.glob('data/graphs/task_*_*.pt'))
print('pt files on disk: %d' % len(files))
g2, f2 = 0, 0
for p in files:
    gs = torch.load(p, weights_only=False)
    g2 += len(gs)
    f2 += sum(g['flow'].y.numel() for g in gs)
print('pt reload: graphs=%d flows=%d' % (g2, f2))
print('match report: %s' % str(g2 == tot_g and f2 == tot_f))
