# from torch_geometric.datasets import Coauthor

# dataset = Coauthor(root='./data', name='CS')
# data = dataset[0]

# print(data)

import scipy.io as sio
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

mat = sio.loadmat("./data/blogcatalog.mat")

adj = mat["network"]   # scipy sparse adjacency matrix (N x N)
labels = mat["group"]  # scipy sparse labels (N x C)
labels = labels[:, 0]  # (N,)
labels_dense = labels.toarray()          # (10312, 1)
labels_dense = labels_dense.squeeze()    # (10312,)
labels_dense = labels_dense.astype(int)

# scipy sparse → NetworkX
G = nx.from_scipy_sparse_matrix(adj)
edge_index = torch.tensor(list(G.edges)).t().contiguous()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(
    edge_index=edge_index,
    embedding_dim=128,   # embedding 维度
    walk_length=80,      # 随机游走长度
    context_size=10,     # Skip-gram 窗口大小
    walks_per_node=20,   # 每个节点随机游走次数
    num_negative_samples=1,
    p=1, q=1,            # p=q=1 → DeepWalk 行为
    sparse=True
).to(device)

# 优化器
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

# 训练
for epoch in tqdm(range(10)):  # epoch 可调
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in model.loader(batch_size=128, shuffle=True):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 获取 embedding
X = model.embedding.weight.data.cpu()  # tensor shape [N, 128]

x = np.array(X, dtype=np.float32)

print("Node features shape:", X.shape)
print("Labels shape:", labels.shape)

data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=edge_index,
    y=torch.from_numpy(labels_dense).long()
)
print(data)

# 保存整个图
torch.save(data, "./data/blogcatalog_data.pt")

# 加载
data_loaded = torch.load("./data/blogcatalog_data.pt")
print(data_loaded)