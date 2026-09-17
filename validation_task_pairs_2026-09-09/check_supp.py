import json
from pathlib import Path

fm = json.load(open('runs/gnn_3layer_residual_s42/forgetting_matrix.json'))
print('FINAL ROW T6: ' + str(fm['6']))
for i in range(1, 7):
    row = fm[str(i)]
    print('trained_up_to %d: %s' % (i, str({k: round(v, 4) for k, v in sorted(row.items(), key=lambda x: int(x[0]))})))

# graph counts
gc = json.load(open('data/graphs/graph_counts.json'))
tot = sum(gc[str(t)][s]['num_graphs'] for t in range(1, 7) for s in ('train', 'val', 'test'))
print('GRAPH TOTAL ON DISK: %d (graph_size=%s benign_ratio=%s)' % (tot, str(gc.get('graph_size')), str(gc.get('benign_ratio'))))
for t in range(1, 7):
    print('task %d: %s' % (t, str({s: gc[str(t)][s]['num_graphs'] for s in ('train', 'val', 'test')})))

# processed totals
import pandas as pd
total = 0
for t in range(1, 7):
    df = pd.read_parquet('data/processed/task_%d.parquet' % t, columns=['canonical_label'])
    total += len(df)
print('PROCESSED TOTAL ROWS: %d' % total)

# eval dir?
print('eval dir exists: ' + str(Path('eval').exists()))
print('data/graphs_ratio2 exists: ' + str(Path('data/graphs_ratio2').exists()))
print('data/graphs_ratio4 exists: ' + str(Path('data/graphs_ratio4').exists()))
