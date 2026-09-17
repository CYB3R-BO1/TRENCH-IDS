import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
from trench_ids.vocab import load_vocab, vocab_fingerprint, vocab_key

vocab = load_vocab(Path('data/graphs/vocab.json'))
print('PROTOCOL size: %d' % len(vocab['PROTOCOL']))
print('L7_PROTO size: %d' % len(vocab['L7_PROTO']))
fp = vocab_fingerprint(vocab)
print('fingerprint: ' + fp)
# Check float-coercion robustness: 6 vs 6.0
print('vocab_key(6)==vocab_key(6.0): ' + str(vocab_key(6) == vocab_key(6.0)))
out = {'protocol_size': len(vocab['PROTOCOL']), 'l7_size': len(vocab['L7_PROTO']), 'fingerprint': fp, 'key_coercion_ok': vocab_key(6) == vocab_key(6.0)}
with open('validation_task_pairs_2026-09-09/05_vocab_validation.json', 'w') as f:
    json.dump(out, f, indent=2)
