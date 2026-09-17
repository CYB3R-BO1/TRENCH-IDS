import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
import torch
from trench_ids import labels
from trench_ids.graphs import LABEL_NAMES, LABEL_LOOKUP

print('LABEL_NAMES: ' + str(LABEL_NAMES))
assert LABEL_NAMES == labels.canonical_classes(), 'LABEL_NAMES mismatch'
locked = {t: sorted(labels.attack_classes_for_task(t)) for t in range(1, 7)}
result = {'label_names': LABEL_NAMES, 'locked': locked, 'tasks': {}, 'mismatches': []}
for t in range(1, 7):
    for split in ('train', 'val', 'test'):
        p = Path('data/graphs/task_%d_%s.pt' % (t, split))
        graphs = torch.load(p, weights_only=False)
        # collect classes across graphs via y
        found_idx = set()
        n_flows = 0
        for g in graphs:
            y = g['flow'].y.tolist()
            found_idx.update(y)
            n_flows += len(y)
        found_names = sorted([LABEL_NAMES[i] for i in sorted(found_idx)])
        attack_found = sorted([c for c in found_names if c != 'Benign'])
        key = '%d_%s' % (t, split)
        result['tasks'][key] = {'n_graphs': len(graphs), 'n_flows': n_flows, 'classes': found_names, 'attack_classes': attack_found}
        if attack_found != locked[t]:
            result['mismatches'].append({'key': key, 'found': attack_found, 'expected': locked[t]})
        # check relations on first graph
        if split == 'train' and graphs:
            ets = sorted([str(e) for e in graphs[0].edge_index_dict.keys()])
            result['tasks'][key]['edge_types'] = ets
            result['tasks'][key]['n_relations'] = len(ets)
            if len(ets) != 11:
                result['mismatches'].append({'key': key, 'n_relations': len(ets)})
        print('%s: graphs=%d flows=%d classes=%s' % (key, len(graphs), n_flows, str(found_names)))
print('mismatches: ' + str(result['mismatches']))
print('MATCHES_LOCKED: ' + str(len(result['mismatches']) == 0))
with open('validation_task_pairs_2026-09-09/06_graphs_validation.json', 'w') as f:
    json.dump(result, f, indent=2)
