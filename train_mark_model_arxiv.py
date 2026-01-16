from model.GCN import build_gnn_model, train_gnn_model
from utility.utils import set_seed, dict_to_argparser
from src.utils import initialize_exp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import torch
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 1. 可调超参数定义
# -------------------------------
params = {
    "dataset_path": f'mark_save/graph_watermarked_arxiv_dim{128}_layer{2}_seed{42}.pt', # 水印图数据路径
    "random_seed": 42,
    "hidden_dim": 128,        # GCN 隐藏维度
    "num_layers": 2,         # GCN 层数，可调消融实验
    "dropout": 0.5,          # Dropout
    "lr": 0.01,              # 学习率
    "weight_decay": 5e-4,    # Adam 权重衰减
    "epochs": 2000,           # 最大训练轮数
    "model_save_path": f"model_save/gcn_mark_arxiv_dim{128}_layer{2}_seed{42}.pth",
    "exp_name": "train_mark_model_arxiv", # 实验名称
    "exp_id": f"mark_gcn_arxiv_dim{128}_layer{2}_seed{42}", # 实验 ID
    "dump_path": "logs", # 日志保存路径
}

params_new = dict_to_argparser(params)
logger = initialize_exp(params_new)

set_seed(params["random_seed"])

# -------------------------------
# 2. 加载 graph 数据
# -------------------------------
sub_data = torch.load(params["dataset_path"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 构建 x 特征：仅替换 node_list 对应的节点
# -------------------------------
x = sub_data["x_orig"].clone().to(device)  # 原始特征
if "node_list" in sub_data and "x_wm" in sub_data:
    node_list = sub_data["node_list"].squeeze()
    x_wm_nodes = sub_data["x_wm"][node_list].to(device)
    x[node_list] = x_wm_nodes  # 仅替换被标记节点的特征

# -------------------------------
# 3. 构建 GPU Data
# -------------------------------
data = Data(
    x=x,
    edge_index=sub_data["edge_index"],
    y=sub_data["y"],
    train_mask=sub_data["train_mask"],
    val_mask=sub_data["val_mask"],
    test_mask=sub_data["test_mask"]
).to(device)

# -------------------------------
# 4. 多层 GCN 定义
# -------------------------------

model = build_gnn_model(params, data).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

# -------------------------------
# 5. 训练循环
# -------------------------------
best_val_acc, best_train_acc, best_test_acc = train_gnn_model(model, data, optimizer, params["epochs"], params["model_save_path"], logger=logger)
logger.info(f"Final Results -- Best Val Acc: {best_val_acc}, Best Train Acc: {best_train_acc}, Best Test Acc: {best_test_acc}")
