import torch
from utility.utils import set_seed
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

set_seed(42)

n_classes, dim = 10, 512  # for class on cifar10 and hidden units on Resnet-18
carriers = torch.randn(n_classes, dim)
carriers /= torch.norm(carriers, dim=1, keepdim=True)
torch.save(carriers, os.path.join(PROJECT_ROOT, "mark_save/carriers.pth"))
