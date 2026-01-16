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

if __name__ == "__main__":
    """
    python train_benign_model_arxiv.py
    """
    # -------------------------------
    # 1. 可调超参数定义
    # -------------------------------

    params = {
        "dataset_path": 'data/ogbn_arxiv_balanced_subgraph.pt',
        "random_seed": 42,
        "hidden_dim": 128,        # GCN 隐藏维度
        "num_layers": 2,         # GCN 层数，可调消融实验
        "dropout": 0.5,          # Dropout
        "lr": 0.01,              # 学习率
        "weight_decay": 5e-4,    # Adam 权重衰减
        "epochs": 2000,           # 最大训练轮数
        "model_save_path": f"model_save/gcn_benign_arxiv_dim{128}_layer{2}_seed{42}.pth",
        "exp_name": "train_benign_model_arxiv", # 实验名称
        "exp_id": f"benign_gcn_arxiv_dim{128}_layer{2}_seed{42}", # 实验 ID
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
    # 3. 构建 GPU Data
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
