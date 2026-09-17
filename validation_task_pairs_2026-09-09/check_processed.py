import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
import pandas as pd
from trench_ids import labels
from trench_ids.preprocess import load_config

cfg = load_config('configs/preprocess.yaml')
print('config out_dir: ' + str(cfg['paths']['out_dir']))
print('attack_per_task: ' + str(cfg['sampling']['attack_per_task']))
print('benign_per_task: ' + str(cfg['sampling']['benign_per_task']))
locked = {t: sorted(labels.attack_classes_for_task(t)) for t in range(1, 7)}
result = {'locked': locked, 'tasks': {}, 'mismatches': []}
for t in range(1, 7):
    p = Path('data/processed/task_%d.parquet' % t)
    df = pd.read_parquet(p, columns=['canonical_label', 'task', 'split', 'source_dataset'])
    classes = sorted([c for c in df['canonical_label'].unique().tolist() if c != 'Benign'])
    task_ids = sorted(df['task'].unique().tolist())
    result['tasks'][str(t)] = {'n_rows': int(len(df)), 'attack_classes': classes, 'task_ids': task_ids, 'class_counts': {k: int(v) for k, v in df['canonical_label'].value_counts().items()}}
    if classes != locked[t]:
        result['mismatches'].append({'task': t, 'found': classes, 'expected': locked[t]})
    if task_ids != [t]:
        result['mismatches'].append({'task': t, 'bad_task_id': task_ids})
    print('T%d: %s rows=%d classes=%s' % (t, str(classes), len(df), str(result['tasks'][str(t)]['class_counts'])))
manifest = json.loads(Path('data/processed/manifest.json').read_text())
result['manifest_themes'] = {k: v['theme'] for k, v in manifest['tasks'].items()}
for k, v in manifest['tasks'].items():
    if v['theme'] != labels.TASK_THEMES[int(k)]:
        result['mismatches'].append({'manifest_theme_mismatch': k})
print('mismatches: ' + str(result['mismatches']))
print('MATCHES_LOCKED: ' + str(len(result['mismatches']) == 0))
with open('validation_task_pairs_2026-09-09/04_processed_validation.json', 'w') as f:
    json.dump(result, f, indent=2)
