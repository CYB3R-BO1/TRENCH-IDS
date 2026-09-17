"""Verify strong_replay_baseline checkpoint loads and runs on GPU."""
import time
from pathlib import Path

import torch

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')

from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import load_split
from trench_ids.labels import canonical_classes
from trench_ids.cl.evaluate import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

graphs_dir = Path('data/graphs')
ckpt_path = Path('runs/strong_replay_baseline/checkpoint_task_6.pt')
print(f'Loading {ckpt_path}...')
t0 = time.time()
model, classifier, ckpt = load_checkpoint(ckpt_path, graphs_dir, device)
print(f'Load time: {time.time() - t0:.1f}s')
print('Model on device:', next(model.parameters()).device)
print('Classifier on device:', next(classifier.parameters()).device)
print('Checkpoint config:', {k: v for k, v in ckpt.get('config', {}).items() if k not in ('label_names',)})

label_names = canonical_classes()
print(f'Label names ({len(label_names)}):', label_names)

# Eval on every task's test split (mimics FINAL_BENCHMARK_RESULTS "final_average_accuracy")
for task_id in range(1, 7):
    test_graphs = load_split(graphs_dir, task_id, 'test')
    print(f'Task {task_id} test: {len(test_graphs)} graphs')
    t0 = time.time()
    result = predict(model, classifier, test_graphs, device, 8, label_names)
    metrics = compute_metrics(result['y_true'], result['y_pred'], label_names)
    print(f'  Task {task_id}: acc={metrics["accuracy"]:.4f} F1-macro={metrics["f1_macro"]:.4f} F1-weighted={metrics["f1_weighted"]:.4f} ({time.time() - t0:.1f}s)')