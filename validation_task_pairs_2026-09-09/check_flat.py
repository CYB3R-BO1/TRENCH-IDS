import sys
sys.path.insert(0, 'src')
from trench_ids.model.flat import FlatFlowEncoder

for name, kw in [('flat', dict(use_host_features=False, mlp_hidden=292)), ('flathost', dict(use_host_features=True, mlp_hidden=254)), ('flathost_small', dict(use_host_features=True, mlp_hidden=123))]:
    m = FlatFlowEncoder(hidden_dim=64, protocol_vocab_size=5, service_vocab_size=239, port_tail_buckets=32, **kw)
    print('%s total=%d' % (name, sum(p.numel() for p in m.parameters())))
