import torch
from utility.utils import set_seed
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    """
    python random_direction_generation.py
    """

    set_seed(42)

    n_classes, dim = 40, 20  # the class number of dataset, the hidden units of final layer in benign model
    carriers = torch.randn(n_classes, dim)
    carriers /= torch.norm(carriers, dim=1, keepdim=True)
    torch.save(carriers, os.path.join(PROJECT_ROOT, f"mark_save/carriers_class{n_classes}_dim{dim}.pth"))
