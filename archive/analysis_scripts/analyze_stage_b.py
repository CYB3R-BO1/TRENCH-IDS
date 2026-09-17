import json
import os

# Seed 42 results
results_42 = {}
for exp in ['normal', 'stability_targeted_by', 'stability_uniform', 'stability_oracle', 
            'reset_targeted_by', 'reset_targeted_by_stability', 'reset_targeted_by_uniform', 'reset_targeted_by_oracle']:
    path = f'runs/stage_b/seed42/t3_t4/{exp}/results.json'
    if os.path.exists(path):
        data = json.load(open(path))
        results_42[exp] = {
            'aulc': data['aulc_macro_f1'],
            'epoch1': data['epoch1_macro_f1'],
            'final': data['final_macro_f1'],
        }

# Seed 1 results
results_1 = {}
for exp in ['normal', 'stability_targeted_by', 'stability_uniform', 'stability_oracle', 
            'reset_targeted_by', 'reset_targeted_by_stability']:
    path = f'runs/stage_b/seed1/t3_t4/{exp}/results.json'
    if os.path.exists(path):
        data = json.load(open(path))
        results_1[exp] = {
            'aulc': data['aulc_macro_f1'],
            'epoch1': data['epoch1_macro_f1'],
            'final': data['final_macro_f1'],
        }

print('=== SEED 42 ===')
for k, v in results_42.items():
    print(f'{k:30s} AULC={v["aulc"]:.4f} Epoch1={v["epoch1"]:.4f} Final={v["final"]:.4f}')

print()
print('=== SEED 1 ===')
for k, v in results_1.items():
    print(f'{k:30s} AULC={v["aulc"]:.4f} Epoch1={v["epoch1"]:.4f} Final={v["final"]:.4f}')

print()
print('=== COMPARISONS (Seed 42) ===')
normal = results_42['normal']['aulc']
print(f'Normal baseline AULC: {normal:.4f}')
print(f'targeted_by stability:   {results_42["stability_targeted_by"]["aulc"]:.4f} (diff={results_42["stability_targeted_by"]["aulc"]-normal:+.4f})')
print(f'uniform stability:       {results_42["stability_uniform"]["aulc"]:.4f} (diff={results_42["stability_uniform"]["aulc"]-normal:+.4f})')
print(f'oracle stability:        {results_42["stability_oracle"]["aulc"]:.4f} (diff={results_42["stability_oracle"]["aulc"]-normal:+.4f})')
print(f'reset targeted_by:       {results_42["reset_targeted_by"]["aulc"]:.4f} (diff={results_42["reset_targeted_by"]["aulc"]-normal:+.4f})')
print(f'reset + targeted_by stab: {results_42["reset_targeted_by_stability"]["aulc"]:.4f} (diff={results_42["reset_targeted_by_stability"]["aulc"]-normal:+.4f})')
print(f'reset + uniform stab:    {results_42["reset_targeted_by_uniform"]["aulc"]:.4f} (diff={results_42["reset_targeted_by_uniform"]["aulc"]-normal:+.4f})')
print(f'reset + oracle stab:     {results_42["reset_targeted_by_oracle"]["aulc"]:.4f} (diff={results_42["reset_targeted_by_oracle"]["aulc"]-normal:+.4f})')

print()
print('=== COMPARISONS (Seed 1) ===')
normal1 = results_1['normal']['aulc']
print(f'Normal baseline AULC: {normal1:.4f}')
print(f'targeted_by stability:   {results_1["stability_targeted_by"]["aulc"]:.4f} (diff={results_1["stability_targeted_by"]["aulc"]-normal1:+.4f})')
print(f'uniform stability:       {results_1["stability_uniform"]["aulc"]:.4f} (diff={results_1["stability_uniform"]["aulc"]-normal1:+.4f})')
print(f'oracle stability:        {results_1["stability_oracle"]["aulc"]:.4f} (diff={results_1["stability_oracle"]["aulc"]-normal1:+.4f})')
print(f'reset targeted_by:       {results_1["reset_targeted_by"]["aulc"]:.4f} (diff={results_1["reset_targeted_by"]["aulc"]-normal1:+.4f})')
print(f'reset + targeted_by stab: {results_1["reset_targeted_by_stability"]["aulc"]:.4f} (diff={results_1["reset_targeted_by_stability"]["aulc"]-normal1:+.4f})')