import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
import torch
from trench_ids.cl.memory_bank import load_memory_bank
from trench_ids import labels

bank = load_memory_bank(Path('runs/gnn_3layer_residual_s42/memory_bank.pt'), map_location='cpu')
print('bank classes: ' + str(sorted(bank.keys())))
print('bank relations (Scanning): ' + str(sorted(bank['Scanning'].keys())))
locked_pairs = {t: sorted(labels.attack_classes_for_task(t)) for t in range(1, 7)}
all_attack = sorted([c for t in range(1, 7) for c in locked_pairs[t]])
print('locked all attack: ' + str(all_attack))
print('bank matches locked set: ' + str(sorted(bank.keys()) == all_attack))
# transferability task files: new classes per task should match locked
for t in range(1, 7):
    tf = json.loads(Path('runs/gnn_3layer_residual_s42/transferability_task_%d.json' % t).read_text())
    print('T%d transferability new classes: %s expected %s' % (t, sorted(tf.keys()), locked_pairs[t]))
with open('validation_task_pairs_2026-09-09/09_bank_validation.json', 'w') as f:
    json.dump({'bank_classes': sorted(bank.keys()), 'matches_locked': sorted(bank.keys()) == all_attack}, f, indent=2)
