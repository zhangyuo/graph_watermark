from model.GCN import MultiLayerGCN
from utility.utils import set_seed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import sys
import os

# -------------------------------
# 1. 可调超参数定义
# -------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
params = {
    "dataset_path": os.path.join(PROJECT_ROOT, 'data/ogbn_arxiv_balanced_subgraph.pt'),
    "random_seed": 42,
    "hidden_dim": 512,        # GCN 隐藏维度
    "num_layers": 2,         # GCN 层数，可调消融实验
    "dropout": 0.5,          # Dropout
    "lr": 0.01,              # 学习率
    "weight_decay": 5e-4,    # Adam 权重衰减
    "epochs": 2000,           # 最大训练轮数
    "model_save_path": os.path.join(PROJECT_ROOT, f"model_save/gcn_benign_arxiv_dim{512}_layer{2}_seed{42}.pth")
}

set_seed(params["random_seed"])

# -------------------------------
# 2. 加载 graph 数据
# -------------------------------
sub_data = torch.load(params["dataset_path"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 5. 构建 GPU Data
# -------------------------------
data = Data(
    x=sub_data["x"],
    edge_index=sub_data["edge_index"],
    y=sub_data["y"],
    train_mask=sub_data["train_mask"],
    val_mask=sub_data["val_mask"],
    test_mask=sub_data["test_mask"]
).to(device)

# -------------------------------
# 6. 多层 GCN 定义
# -------------------------------

num_features = data.num_node_features
num_classes = int(data.y.max().item() + 1)

model = MultiLayerGCN(
    in_channels=num_features,
    hidden_channels=params["hidden_dim"],
    out_channels=num_classes,
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

# -------------------------------
# 7. 训练函数
# -------------------------------


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# -------------------------------
# 8. 测试函数
# -------------------------------


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]).float().mean().item()
    test_acc  = (pred[data.test_mask]  == data.y[data.test_mask]).float().mean().item()
    return train_acc, val_acc, test_acc


# -------------------------------
# 9. 训练循环
# -------------------------------
best_val_acc = 0
for epoch in range(1, params["epochs"]+1):
    loss = train()
    train_acc, val_acc, test_acc = test()

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), params["model_save_path"])

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

print("Training finished! Best validation accuracy:", best_val_acc)
