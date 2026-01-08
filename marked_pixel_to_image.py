import numpy as np
from PIL import Image

marked = np.load("./cifar10_marked/29.npy")
marked = np.clip(marked * 255, 0, 255).astype(np.uint8)  # 如果是 0~1 的 float
Image.fromarray(marked).save("./cifar10_marked/29.png")


marked = np.load("./cifar10_marked/77.npy")
marked = np.clip(marked * 255, 0, 255).astype(np.uint8)  # 如果是 0~1 的 float
Image.fromarray(marked).save("./cifar10_marked/77.png")