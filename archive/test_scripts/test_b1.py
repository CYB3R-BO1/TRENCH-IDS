import json
from pathlib import Path
from trench_ids.cl.device import resolve_device
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import load_split
from trench_ids.labels import canonical_classes
from trench_ids.cl.evaluate import compute_metrics

device = resolve_device('cuda')
checkpoint_path = Path("runs/strong_replay_baseline/checkpoint_task_1.pt")
model, classifier, _ = load_checkpoint(checkpoint_path, Path("data/graphs"), resolve_device("cuda"))
model.eval()

from trench_ids.cl.train import load_split
from trench_ids.labels import canonical_classes
from trench_ids.cl.inference import predict
from trench_ids.cl.evaluate import compute_metrics

target_graphs = load_split(Path("data/graphs"), 2, "test")
result = predict(model, classifier, target_graphs, resolve_device("cuda"), 8, canonical_classes())
metrics = compute_metrics(result["y_true"], result["y_pred"], canonical_classes())
print(f"Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")