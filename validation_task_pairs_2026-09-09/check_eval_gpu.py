import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
import torch
from trench_ids.cl.device import resolve_device
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.evaluate import compute_metrics
from trench_ids.cl.train import load_split, average_forgetting, final_average_accuracy
from trench_ids import labels

device = resolve_device('auto')
print('device: ' + str(device))
run_dir = Path('runs/gnn_3layer_residual_s42')
graphs_dir = Path('data/graphs')
model, classifier, ckpt = load_checkpoint(run_dir / 'checkpoint_task_6.pt', graphs_dir, device)
label_names = ckpt['config']['label_names']
print('label_names: ' + str(label_names))
print('model cfg: layers=%s residual=%s' % (str(ckpt['config'].get('num_layers')), str(ckpt['config'].get('use_residual'))))
locked = {t: sorted(labels.attack_classes_for_task(t)) for t in range(1, 7)}
per_task = {}
for t in range(1, 7):
    graphs = load_split(graphs_dir, t, 'test')
    res = predict(model, classifier, graphs, device, 8, label_names)
    m = compute_metrics(res['y_true'], res['y_pred'], label_names)
    # true attack classes present
    true_idx = sorted(set(res['y_true']))
    true_names = sorted([label_names[i] for i in true_idx])
    per_task[str(t)] = {'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'], 'true_classes': true_names, 'expected_attack': locked[t]}
    print('T%d acc=%.4f f1=%.4f true=%s expected_attack=%s' % (t, m['accuracy'], m['f1_macro'], str(true_names), str(locked[t])))
fm = json.loads((run_dir / 'forgetting_matrix.json').read_text())
print('stored forgetting: %.6f acc: %.6f' % (json.loads((run_dir / 'summary.json').read_text())['average_forgetting'], json.loads((run_dir / 'summary.json').read_text())['final_average_accuracy']))
print('recomputed forgetting: %.6f final_acc: %.6f' % (average_forgetting({int(k): {int(kk): vv for kk, vv in v.items()} for k, v in fm.items()}), final_average_accuracy({int(k): {int(kk): vv for kk, vv in v.items()} for k, v in fm.items()})))
with open('validation_task_pairs_2026-09-09/08_eval_gpu.json', 'w') as f:
    json.dump({'device': str(device), 'label_names': label_names, 'per_task': per_task}, f, indent=2)
