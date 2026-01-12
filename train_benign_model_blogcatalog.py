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
    # 保存好的 embedding
    "embedding_file": os.path.join(PROJECT_ROOT, 'data/blogcatalog_data.pt'),
    "random_seed": 42,
    "hidden_dim": 512,        # GCN 隐藏维度
    "num_layers": 2,         # GCN 层数，可调消融实验
    "dropout": 0.5,          # Dropout
    "lr": 0.01,              # 学习率
    "weight_decay": 5e-4,    # Adam 权重衰减
    "epochs": 200,           # 最大训练轮数
    "train_ratio": 0.6,
    "val_ratio": 0.2,
    "test_ratio": 0.2,
    "model_save_path": os.path.join(PROJECT_ROOT, f"model_save/gcn_benign_model_dim{512}_layer{2}_seed{42}.pth")
}

set_seed(params["random_seed"])

# -------------------------------
# 2. 加载 embedding 数据
# -------------------------------
data = torch.load(params["embedding_file"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 3. 标签处理（取第一列）
# -------------------------------
labels = data.y
# labels = labels.flatten()  # shape (N,)
labels = labels.to(torch.long)

# -------------------------------
# 4. 划分训练/验证/测试集
# -------------------------------
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
np.random.shuffle(indices)

train_idx = torch.tensor(
    indices[:int(params["train_ratio"]*num_nodes)], dtype=torch.long)
val_idx = torch.tensor(indices[int(params["train_ratio"]*num_nodes):int(
    (params["train_ratio"]+params["val_ratio"])*num_nodes)], dtype=torch.long)
test_idx = torch.tensor(indices[int(
    (params["train_ratio"]+params["val_ratio"])*num_nodes):], dtype=torch.long)
# indices = np.arange(num_nodes)
# train_idx, test_val_idx = train_test_split(indices, stratify=labels.cpu().numpy(), train_size=0.6, random_state=params["random_seed"])
# val_idx, test_idx = train_test_split(test_val_idx, stratify=labels[test_val_idx].cpu().numpy(), test_size=0.5, random_state=params["random_seed"])

# -------------------------------
# 5. 构建 GPU Data
# -------------------------------
data = Data(
    x=data.x.to(device),
    edge_index=data.edge_index.to(device),
    y=labels.to(device)
)

# -------------------------------
# 6. 多层 GCN 定义
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
    model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

# -------------------------------
# 7. 训练函数
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
# 8. 测试函数
# -------------------------------


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = (pred[train_idx] == data.y[train_idx]
                 ).sum().item() / train_idx.size(0)
    val_acc = (pred[val_idx] == data.y[val_idx]).sum().item() / val_idx.size(0)
    test_acc = (pred[test_idx] == data.y[test_idx]
                ).sum().item() / test_idx.size(0)
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
