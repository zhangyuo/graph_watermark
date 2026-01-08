import torch

torch.save({
  'type': 'per_sample',
  'content': {
    29: 'cifar10_marked/29.npy',
    77: 'cifar10_marked/77.npy',
  }
}, "radioactive_data.pth")
