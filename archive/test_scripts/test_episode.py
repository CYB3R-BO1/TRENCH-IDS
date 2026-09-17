import sys
sys.path.insert(0, 'src')
from pathlib import Path
from trench_ids.cl.episode_generator import EpisodeGenerator

generator = EpisodeGenerator(Path('data/graphs'))
generator.initialize()
meta_train, meta_val, _ = generator.generate_episodes(n_train=1, n_val=1)

ep = meta_train[0]
print('Keys:', list(ep.keys()))
for k, v in ep.items():
    ln = len(v) if hasattr(v, '__len__') else 'N/A'
    print(f'  {k}: len={ln}')