import torch

print('CUDA:', torch.cuda.is_available())
graphs = torch.load('data/no-cap/graphs/task_1_train.pt', weights_only=False, map_location='cpu')
print(f'Loaded {len(graphs)} graphs to CPU')
print(f'First graph flow x: {graphs[0]["flow"].x.shape}')
print(f'First graph flow x device: {graphs[0]["flow"].x.device}')