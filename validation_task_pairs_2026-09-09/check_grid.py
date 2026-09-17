import json
fm = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(open('runs/gnn_3layer_residual_s42/forgetting_matrix.json')).items()}
for A in range(1, 7):
    row = []
    for i in range(1, 7):
        if i > A:
            row.append('--')
        elif i == A:
            row.append('0.000')
        else:
            peak = max(fm[t][i] for t in range(i, A))
            row.append('%.3f' % (peak - fm[A][i]))
    print('after T%d: %s' % (A, ' '.join(row)))
