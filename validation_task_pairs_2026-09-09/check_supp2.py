import json
from pathlib import Path

for r in ['gnn_3layer_residual_s42', 'gnn_3layer_residual_s1', 'gnn_3layer_residual_s2', 'flathost_replay_s42', 'flathost_replay_s1', 'flathost_replay_s2']:
    p = Path('runs/%s/summary.json' % r)
    if p.exists():
        s = json.load(open(p))
        print('%s: forget=%.4f acc=%.4f' % (r, s['average_forgetting'], s['final_average_accuracy']))
    else:
        print('%s: MISSING' % r)

# ablation runs referenced?
for r in ['baseline_test', 'final_baseline', 'fact_strong']:
    print(r, Path('runs/%s' % r).exists())

# b1 / intervention / budget dirs?
for r in ['b1_loto', 'transfer_intervention', 'runs/transfer_predictor']:
    print(r, Path(r).exists())

# per-class pooled metrics?
print('eval/pooled exists:', Path('eval/pooled_final_metrics.json').exists())
print('strong_replay_baseline exists:', Path('runs/strong_replay_baseline').exists())
