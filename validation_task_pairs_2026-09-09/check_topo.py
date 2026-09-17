import sys
sys.path.insert(0, 'src')
from pathlib import Path
import torch
import numpy as np

for t in range(1, 7):
    graphs = torch.load(Path('data/graphs/task_%d_train.pt' % t), weights_only=False)
    flows, hosts, ports, nodes, edges, hdeg = [], [], [], [], [], []
    for g in graphs:
        nf = g['flow'].x.shape[0]
        nh = int(g['host'].num_nodes)
        np_ = int(g['port'].num_nodes)
        nn = nf + nh + np_ + int(g['protocol'].num_nodes) + int(g['service'].num_nodes)
        ne = sum(int(g[e].edge_index.shape[1]) for e in g.edge_index_dict.keys())
        flows.append(nf)
        hosts.append(nh)
        ports.append(np_)
        nodes.append(nn)
        edges.append(ne)
        hdeg.append(float(g['host'].x[:, 0].mean()))
    print('T%d n=%d flow=%.1f host=%.1f port=%.1f nodes=%.1f edges=%.1f hostdeg=%.1f' % (t, len(graphs), float(np.mean(flows)), float(np.mean(hosts)), float(np.mean(ports)), float(np.mean(nodes)), float(np.mean(edges)), float(np.mean(hdeg))))
