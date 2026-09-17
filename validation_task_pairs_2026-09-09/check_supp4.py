import sys
sys.path.insert(0, 'src')
from pathlib import Path
import torch
from trench_ids.vocab import load_vocab
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN

vocab = load_vocab(Path('data/graphs/vocab.json'))
graphs = torch.load(Path('data/graphs/task_1_train.pt'), weights_only=False)
g = graphs[0]
for nl in (1, 2, 3):
    m = RelationSpecificHeteroGNN.from_graph(g, hidden_dim=64, protocol_vocab_size=len(vocab['PROTOCOL']), service_vocab_size=len(vocab['L7_PROTO']), num_layers=nl, attn_dim=128, port_tail_buckets=32, fusion='attention', use_residual=(nl > 1))
    enc = sum(p.numel() for n, p in m.named_parameters() if n.startswith('encoders.'))
    conv = sum(p.numel() for n, p in m.named_parameters() if '.conv.' in n)
    fus = sum(p.numel() for n, p in m.named_parameters() if '.fusion.' in n)
    tot = sum(p.numel() for p in m.parameters())
    print('layers=%d total=%d enc=%d conv=%d fus=%d residual=%s' % (nl, tot, enc, conv, fus, str(nl > 1)))
