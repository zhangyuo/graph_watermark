import numpy as np
import os

base = "./cifar10_classes"
os.makedirs(base, exist_ok=True)

# for n_cl in [10, 20, 50, 100, 200, 500]:
for n_cl in [10]:
    np.save(f"{base}/{n_cl}.npy", np.arange(n_cl))
