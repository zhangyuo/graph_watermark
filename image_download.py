import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

# # 设置转换（预处理）
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])


# mkdir -p data/cifar10
# cd data/cifar10

# wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# tar -xvzf cifar-10-python.tar.gz


# # 下载训练集
# trainset = CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)

# # 下载测试集
# testset = CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)

# print(len(trainset), len(testset))

# 1. 加载 CIFAR-10（不需要 transform）
dataset = CIFAR10(
    root='./data/cifar10',
    train=True,
    download=False
)

# 2. 导出目录
out_root = './cifar10_images'
os.makedirs(out_root, exist_ok=True)

# 3. 每个 class 一个子目录（ImageFolder 兼容）
for cls in range(10):
    os.makedirs(os.path.join(out_root, str(cls)), exist_ok=True)

# 4. 导出前 N 张图片（比如 100 张先测试）
N = 100

for i in range(N):
    img, label = dataset[i]        # img 是 PIL Image
    save_path = os.path.join(out_root, str(label), f'{i}.png')
    img.save(save_path)

print("Done exporting images.")