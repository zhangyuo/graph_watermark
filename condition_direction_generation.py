import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from model.GCN import build_gnn_model
from os.path import basename, join
import os
from utility.utils import set_seed
from mark_embed import build_x_sub_patched
import numpy as np
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")


# ---------------- 配置 ----------------
seed_num = 42
set_seed(seed_num)
n_classes = 40
nodes_per_class = 100
embedding_dim = 512
gnn_layers = 2
dropout = 0.5
perturb_amplitude = 1.0
epochs = 200
lambda_x = 1.0
lambda_ft = 1.0
lambda_cos = 1.0
lr = 0.01
dataset_path = f"data/ogbn_arxiv_balanced_subgraph.pt"
benign_model_path = f"model_save/gcn_benign_arxiv_dim{embedding_dim}_layer{gnn_layers}_seed{seed_num}.pth"
carriers_save_path = f"mark_save/carriers_class{n_classes}_dim{embedding_dim}_uc.pth"
save_mark_dataset_path = f"mark_save/graph_watermarked_arxiv_dim{embedding_dim}_layer{gnn_layers}_seed{seed_num}.pt"

# ---------------- 加载图数据 ----------------
loaded = torch.load(dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Data(
            x=loaded["x"],
            edge_index=loaded["edge_index"],
            y=loaded["y"],
            train_mask=loaded.get("train_mask"),
            val_mask=loaded.get("val_mask"),
            test_mask=loaded.get("test_mask")
        )
data = data.to(device)
print(f"Using device: {device}")

# ---------------- 加载模型 ----------------
model = build_gnn_model(params={"hidden_dim":embedding_dim, "num_layers":gnn_layers, "dropout":dropout}, data=data)
ckpt = torch.load(benign_model_path)
model.load_state_dict(ckpt)
model.eval().to(device)

# Load gnn mooel classifier weight
print(f"model layer keys: {ckpt.keys()}")
last_conv = model.convs[-1]
W_origin = last_conv.lin.weight.detach()  # shape: (C, D_hidden)

x_wm = data.x.clone()

preset_directions = torch.randn(n_classes, embedding_dim)
preset_directions /= torch.norm(preset_directions, dim=1, keepdim=True)

# ---------------- 阶段 1: 每个类别生成 u_c ----------------
all_nodes_num = 0
carriers = []
cos_with_delta_phi = []
cos_with_classifier = []
mark_node_list = []
cos_upper_list = []

for c in tqdm(range(n_classes)):
    # if c > 0:
    #     continue
    print(f"\n=== Processing class {c} ===")
    # 1. 每个类别随机采样节点
    idx_all = (data.y == c).nonzero(as_tuple=True)[0]
    node_ids = idx_all[torch.randperm(len(idx_all))[:nodes_per_class]]
    all_nodes_num += node_ids.size(0)

    # get subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=gnn_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    x_sub = data.x[subset]  # node features of the subgraph nodes
    x_nodes_orig = x_sub[mapping].clone().detach()
    x_nodes = x_nodes_orig.clone().detach().to(device).requires_grad_(True)

    # ---------------- 白化空间初始化 Δφ ----------------
    with torch.no_grad():
        ft_orig = model.encode(x_sub, edge_index_sub)[mapping]  # 原始 embedding

        direction = preset_directions[c].to(device)

        # # 随机微扰 ΔX_init
        # delta_x_init = (torch.randn_like(x_nodes) * 1e-3)
        # x_nodes_perturbed = x_nodes_orig + delta_x_init

        # x_sub_patched = build_x_sub_patched(x_sub, mapping, x_nodes_perturbed)
        # ft_perturbed = model.encode(x_sub_patched, edge_index_sub)[mapping]

        # delta_phi = ft_perturbed - ft_orig  # [nodes_per_class, embedding_dim]

        # # 中心化
        # delta_phi_centered = delta_phi - delta_phi.mean(dim=0, keepdim=True)
        # # 协方差矩阵
        # cov = delta_phi_centered.T @ delta_phi_centered / (delta_phi_centered.shape[0] - 1)
        # # SVD 白化
        # U, S, _ = torch.linalg.svd(cov)
        # W = U @ torch.diag(1.0 / (S.sqrt() + 1e-8)) @ U.T
        # ft_whitened = delta_phi_centered @ W
        # # 类均值方向
        # mu_c_whitened = ft_whitened.mean(dim=0)
        # v = mu_c_whitened / (mu_c_whitened.norm() + 1e-8)
        # direction = v

        # # 构造其正交补空间
        # P_orth = torch.eye(v.numel(), device=v.device) - torch.outer(v, v)
        # delta_phi_orth = ft_whitened @ P_orth

        # # SVD 得到主方向
        # _, _, Vh = torch.linalg.svd(delta_phi_orth, full_matrices=False)
        # u_c = Vh[0]
        # direction = u_c / (u_c.norm() + 1e-8)

        # 计算cosine similarity的理论上界
        num_jvp_samples = 256   # 128~512 都可以
        eps = 1e-3              # 小扰动，保证一阶近似

        delta_phi_list = []

        for _ in range(num_jvp_samples):
            delta_x = torch.randn_like(x_nodes_orig)
            delta_x = delta_x / (delta_x.norm(dim=1, keepdim=True) + 1e-8)
            delta_x = eps * delta_x

            x_nodes_perturbed = x_nodes_orig + delta_x
            x_sub_perturbed = build_x_sub_patched(x_sub, mapping, x_nodes_perturbed)

            ft_perturbed = model.encode(x_sub_perturbed, edge_index_sub)[mapping]
            delta_phi = ft_perturbed - ft_orig        # [nodes_per_class, D]

            # 🔴 关键：聚合成一个 embedding-level Δφ
            delta_phi_mean = delta_phi.mean(dim=0)    # [D]
            delta_phi_list.append(delta_phi_mean)

        DeltaPhi = torch.stack(delta_phi_list, dim=0)   # [M, D]
        DeltaPhi = DeltaPhi - DeltaPhi.mean(dim=0, keepdim=True)

        # SVD
        U, S, Vh = torch.linalg.svd(DeltaPhi, full_matrices=False)

        # 取前 k 个主方向
        k = min(32, Vh.shape[0])   # 32 / 64 都行
        V_k = Vh[:k].T             # [D, k]

        u = direction / direction.norm()

        u_proj = V_k @ (V_k.T @ u)
        cos_upper = torch.nn.functional.cosine_similarity(
            u_proj.unsqueeze(0),
            u.unsqueeze(0),
            dim=1
        ).item()

        cos_upper_list.append(cos_upper)
        print(f"[Class {c}] Theoretical cosine upper bound: {cos_upper:.4f}")

    print("Initialized class direction norm:", direction.norm().item())

    # ---------------- SGD 优化 ΔX 沿固定方向 ----------------
    optimizer = optim.SGD([x_nodes], lr=lr)
    for epoch in range(epochs): 
        optimizer.zero_grad()

        # x_sub = data.x.clone()
        # x_sub[node_ids] = x_nodes
        # ft_new = model.encode(x_sub, data.edge_index)[node_ids]
        x_sub_patched = build_x_sub_patched(x_sub, mapping, x_nodes)
        ft_new = model.encode(x_sub_patched, edge_index_sub)[mapping]
        
        delta_phi = ft_new - ft_orig

        # loss: 对齐方向 + 正则
        # loss_align = -torch.sum(delta_phi * direction)
        # proj = torch.sum(delta_phi * direction, dim=1)
        # # loss_align = - torch.mean(proj * torch.relu(proj))
        # theta = torch.tensor(math.pi / 3)   # 60°
        # delta_norm = torch.norm(delta_phi, dim=1)
        # loss_cone = torch.mean(torch.relu(
        #     torch.cos(theta) * delta_norm - proj
        # ))
        # loss_align = - torch.mean(proj)
        # # loss_reg = lambda_x * torch.norm(x_nodes - x_nodes_orig, dim=1).mean()
        # # loss_ft = lambda_ft * torch.norm(ft_new - ft_orig, dim=1).mean()
        # # cos_d = torch.nn.functional.cosine_similarity(delta_phi, direction, dim=1)
        # # loss_cos  = lambda_cos * cos_d.mean()
        # loss = loss_cone
        proj = torch.sum(delta_phi * direction, dim=1)
        mask = proj > -10000
        delta_phi = delta_phi[mask]
        loss_align = -torch.sum(delta_phi * direction)
        loss = loss_align

        loss.backward()
        optimizer.step()

        # L∞ 投影
        # with torch.no_grad():
        #     x_nodes.copy_(torch.clamp(x_nodes, x_nodes_orig - perturb_amplitude,
        #                               x_nodes_orig + perturb_amplitude))

        # ---------------- 打印迭代信息 ----------------
        if epoch % 10 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                cos_sim = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
                alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
                alpha_std = torch.sum(delta_phi * direction, dim=1).std().item()
                ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
                x_l2_dist = torch.norm(x_nodes[mask] - x_nodes_orig[mask], dim=1).mean().item()
                x_max_amplitude = (x_nodes[mask] - x_nodes_orig[mask]).abs().max().item()
                print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | "
                    f"alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
                    f"cosine_mean: {cos_sim.mean().item():.6f} | cosine_std: {cos_sim.std().item():.6f} | "
                    f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f}")

    # ---------------- 最终 Δφ SVD -> u_c ----------------
    with torch.no_grad():
        x_sub_final = build_x_sub_patched(x_sub, mapping, x_nodes)
        ft_new = model.encode(x_sub_final, edge_index_sub)[mapping]
        delta_phi = ft_new - ft_orig  # [nodes_per_class, embedding_dim]

        # SVD
        U, S, Vh = torch.linalg.svd(delta_phi, full_matrices=False)
        u_c = Vh[0]  # 主方向
        u_c /= torch.norm(u_c)
        print(f"\nClass {c} final Δφ SVD completed:")
        print(f"Singular values S: {S.cpu().numpy()}")
        explained_ratio = S**2 / (S**2).sum()
        print(f"Explained variance ratio: {explained_ratio.cpu().numpy()}")
        print(f"Cumulative explained variance: {torch.cumsum(explained_ratio, dim=0).cpu().numpy()}")

        # u_c = direction
        carriers.append(u_c)

        mask = torch.sum(delta_phi * u_c.unsqueeze(0), dim=1) > -10000
        delta_phi = delta_phi[mask]

        # ---------------- 打印最终信息 ----------------
        print(f"Generated u_c norm: {u_c.norm().item()}")
        print(f"u_c shape: {u_c.shape}")
        print(f"First 10 dimensions of u_c: {u_c[:10].cpu().numpy()}\n")
        print(f"last 10 dimensions of u_c: {u_c[-10:].cpu().numpy()}\n")
        alpha_mean = torch.sum(delta_phi * u_c.unsqueeze(0), dim=1).mean().item()
        alpha_std = torch.sum(delta_phi * u_c.unsqueeze(0), dim=1).std().item()
        cos_phi = torch.nn.functional.cosine_similarity(delta_phi, u_c.unsqueeze(0), dim=1)
        ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
        x_l2_dist = torch.norm(x_nodes[mask] - x_nodes_orig[mask], dim=1).mean().item()
        x_max_amplitude = (x_nodes[mask] - x_nodes_orig[mask]).abs().max().item()
        print(f"Final alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
              f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item()} | "
              f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f}\n")
        cos_with_delta_phi.append(cos_phi.mean().item())
        print(f"class {c} cosine similarity with delta phi: {cos_phi.cpu().numpy()}")
        print(f"class {c} mean cosine similarity with delta phi: {cos_phi.mean().item():.6f}")

        W_c = W_origin[c]  # classifier weight for class c
        cos_w = torch.nn.functional.cosine_similarity(u_c.unsqueeze(0), W_c.unsqueeze(0), dim=1).item()
        cos_with_classifier.append(cos_w)
        print(f"Cosine similarity between u_c and classifier weight W_c: {cos_w:.6f}")

        x_nodes_tensor = x_nodes[mask].detach()
        node_ids_tensor = node_ids[mask]
        x_wm[node_ids_tensor] = x_nodes_tensor.detach()
        mark_node_list.extend(node_ids_tensor.cpu().tolist())

# ---------------- 保存 carriers ----------------
carriers = torch.stack(carriers, dim=0)
carriers = carriers.to('cpu')
torch.save(carriers, carriers_save_path)
print(f"\nSaved {n_classes} carriers (u_c) to {carriers_save_path}")
# ---------------- 保存带水印的特征数据集 ----------------
torch.save({
            "x_orig": data.x.cpu(),
            "x_wm": x_wm.cpu(),
            "edge_index": data.edge_index.cpu(),
            "y": data.y.cpu(),
            "train_mask": data.train_mask.cpu(),
            "val_mask": data.val_mask.cpu(),
            "test_mask": data.test_mask.cpu(),
            "node_list": torch.tensor(mark_node_list, dtype=torch.long).cpu()
        }, save_mark_dataset_path)
print(f"Saved watermarked dataset to {save_mark_dataset_path}")
print(f"Total marked nodes: {len(mark_node_list)}")
print(f"All sampled nodes: {all_nodes_num}")
# ---------------- 打印整体信息 ----------------
print(f"\nCosine upper bounds for all classes: {cos_upper_list}")
print(f"\nOverall cosine similarities between u_c and delta phi:\n {cos_with_delta_phi}")
print(f"\nOverall cosine similarities between u_c and classifier weights:\n {cos_with_classifier}")
print(f"\nOverall mean cosine similarity with delta phi: {np.mean(cos_with_delta_phi):.6f}")
print(f"Overall mean cosine similarity with classifier weight: {np.mean(cos_with_classifier):.6f}")
print(f"Overall stddev cosine similarity with classifier weight: {np.std(cos_with_classifier):.6f}")
print("Watermarking process completed.")
