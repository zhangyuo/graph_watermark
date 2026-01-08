import torch
from torchvision import models

# resnet18 = models.resnet18(pretrained=True)  # ImageNet-1k
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 10)  # CIFAR-10

torch.save({
    "model": resnet18.state_dict(),
    "params": {
      "architecture": "resnet18",
      "num_classes": 10
    }
  }, "pretrained_resnet18.pth")