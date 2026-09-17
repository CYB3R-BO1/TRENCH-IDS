import torch
from torch_geometric.loader import DataLoader

print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))

graphs = torch.load('data/no-cap/graphs/task_1_train.pt', weights_only=False, map_location='cuda')
print(f'Loaded {len(graphs)} graphs to GPU')

loader = DataLoader(graphs[:4], batch_size=2)
batch = next(iter(loader))
device = batch['flow'].x.device
print(f'Batch loaded to {device}')
print(f'Flow x: {batch["flow"].x.shape}')
print(f'Flow y: {batch["flow"].y.shape}')