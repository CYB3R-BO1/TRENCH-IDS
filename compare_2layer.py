import json
import numpy as np

# Read all three seeds for 2-layer
for seed in [42, 1, 2]:
    with open(f'runs/gnn_2layer_residual_s{seed}/summary.json') as f:
        s = json.load(f)
    print('Seed {}: forgetting={:.4f}, acc={:.4f}'.format(seed, s['average_forgetting'], s['final_average_accuracy']))

print()

# 2-layer stats
forget = []
acc = []
for seed in [42, 1, 2]:
    with open(f'runs/gnn_2layer_residual_s{seed}/summary.json') as f:
        s = json.load(f)
    forget.append(s['average_forgetting'])
    acc.append(s['final_average_accuracy'])
print('2-layer GNN + residual + replay:')
print('  Forgetting: {:.4f} +- {:.4f}'.format(np.mean(forget), np.std(forget)))
print('  Accuracy:   {:.4f} +- {:.4f}'.format(np.mean(acc), np.std(acc)))

# 3-layer stats
forget3 = [0.089190, 0.082315, 0.070002]
acc3 = [0.895734, 0.903800, 0.907552]
print('3-layer GNN + residual + replay:')
print('  Forgetting: {:.4f} +- {:.4f}'.format(np.mean(forget3), np.std(forget3)))
print('  Accuracy:   {:.4f} +- {:.4f}'.format(np.mean(acc3), np.std(acc3)))

# Flat+Host stats
forget_f = [0.091070, 0.094894, 0.129020]
acc_f = [0.886854, 0.879426, 0.850277]
print('Flat+Host + replay:')
print('  Forgetting: {:.4f} +- {:.4f}'.format(np.mean(forget_f), np.std(forget_f)))
print('  Accuracy:   {:.4f} +- {:.4f}'.format(np.mean(acc_f), np.std(acc_f)))

print()
# Paired diffs 2-layer vs Flat+Host
print('Paired diffs (2-layer GNN vs Flat+Host):')
for seed in [42, 1, 2]:
    with open('runs/gnn_2layer_residual_s{}/summary.json'.format(seed)) as f:
        gnn = json.load(f)
    with open('runs/flathost_replay_s{}/summary.json'.format(seed)) as f:
        flat = json.load(f)
    fd = gnn['average_forgetting'] - flat['average_forgetting']
    ad = gnn['final_average_accuracy'] - flat['final_average_accuracy']
    print('  Seed {}: forgetting {:+.4f}, acc {:+.4f}'.format(seed, fd, ad))

print()
# Paired diffs 3-layer vs Flat+Host
print('Paired diffs (3-layer GNN vs Flat+Host):')
seeds = [42, 1, 2]
for i, seed in enumerate(seeds):
    with open('runs/flathost_replay_s{}/summary.json'.format(seed)) as f:
        flat = json.load(f)
    fd = forget3[i] - flat['average_forgetting']
    ad = acc3[i] - flat['final_average_accuracy']
    print('  Seed {}: forgetting {:+.4f}, acc {:+.4f}'.format(seed, fd, ad))