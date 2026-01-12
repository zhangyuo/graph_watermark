from model.GCN import MultiLayerGCN
from utility.utils import set_seed
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import os

# -------------------------------
# 1. Hyperparameters
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
params = {
    # [WM] load watermarked graph instead of benign embedding
    "wm_graph_file": os.path.join(PROJECT_ROOT, "mark_save/graph_watermarked.pt"),

    "random_seed": 42,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,

    # [WM] save as watermarked model
    "model_save_path": os.path.join(
        PROJECT_ROOT,
        f"model_save/gcn_mark_model_dim{512}_layer{2}_seed{42}.pth"
    )
}

set_seed(params["random_seed"])

# -------------------------------
# 2. Load watermarked graph
# -------------------------------
ckpt = torch.load(params["wm_graph_file"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

x = ckpt["x_wm"]           # [WM] use watermarked node features
edge_index = ckpt["edge_index"]
labels = ckpt["y"].to(torch.long)

num_nodes = x.size(0)

# -------------------------------
# 3. Train / Val / Test split
# -------------------------------
indices = np.arange(num_nodes)
np.random.shuffle(indices)

train_idx = torch.tensor(
    indices[:int(params["train_ratio"] * num_nodes)], dtype=torch.long
)
val_idx = torch.tensor(
    indices[int(params["train_ratio"] * num_nodes):
            int((params["train_ratio"] + params["val_ratio"]) * num_nodes)],
    dtype=torch.long
)
test_idx = torch.tensor(
    indices[int((params["train_ratio"] + params["val_ratio"]) * num_nodes):],
    dtype=torch.long
)

# -------------------------------
# 4. Build PyG Data
# -------------------------------
data = Data(
    x=x.to(device),
    edge_index=edge_index.to(device),
    y=labels.to(device)
)

# -------------------------------
# 5. Build model (same as benign)
# -------------------------------
num_features = data.num_node_features
num_classes = int(labels.max().item() + 1)

model = MultiLayerGCN(
    in_channels=num_features,
    hidden_channels=params["hidden_dim"],
    out_channels=num_classes,
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params["lr"],
    weight_decay=params["weight_decay"]
)

# -------------------------------
# 6. Train function
# -------------------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

# -------------------------------
# 7. Evaluation function
# -------------------------------
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_acc = (pred[train_idx] == data.y[train_idx]).float().mean().item()
    val_acc = (pred[val_idx] == data.y[val_idx]).float().mean().item()
    test_acc = (pred[test_idx] == data.y[test_idx]).float().mean().item()

    return train_acc, val_acc, test_acc

# -------------------------------
# 8. Training loop
# -------------------------------
best_val_acc = 0.0

for epoch in range(1, params["epochs"] + 1):
    loss = train()
    train_acc, val_acc, test_acc = test()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), params["model_save_path"])

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f} | "
            f"Test {test_acc:.4f}"
        )

print("Watermarked model training finished.")
print("Best validation accuracy:", best_val_acc)
