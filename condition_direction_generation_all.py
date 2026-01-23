import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from model.GCN import build_gnn_model
from utility.utils import set_seed
from mark_embed import build_x_sub_patched
import numpy as np
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings("ignore")

# ---------------- 配置 ----------------
seed_num = 42
set_seed(seed_num)
n_classes = 40
nodes_per_class = 100
embedding_dim = 512
gnn_layers = 2
dropout = 0.5
epochs = 1000
max_adam_lr = 0.01 # [0.001]
lr = 0.1 # [0.01]
momentum = 0.9
perturb_amplitude = 5.0 # [0.1]
init_clamp_max_perturb = 1.0 # [1e-3]
dataset_path = f"data/ogbn_arxiv_balanced_subgraph.pt"
benign_model_path = f"model_save/gcn_benign_arxiv_dim{embedding_dim}_layer{gnn_layers}_seed{seed_num}.pth"
carriers_save_path = f"mark_save/carriers_class{n_classes}_dim{embedding_dim}_uc.pth"
save_mark_dataset_path = f"mark_save/graph_watermarked_arxiv_dim{embedding_dim}_layer{gnn_layers}_seed{seed_num}.pt"

# ---------------- 加载图数据 ----------------
loaded = torch.load(dataset_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Data(
    x=loaded["x"].to(device),
    edge_index=loaded["edge_index"].to(device),
    y=loaded["y"].to(device),
    train_mask=loaded.get("train_mask"),
    val_mask=loaded.get("val_mask"),
    test_mask=loaded.get("test_mask")
)
print(f"Using device: {device}")

# ---------------- 加载模型 ----------------
model = build_gnn_model(
    params={"hidden_dim": embedding_dim, "num_layers": gnn_layers, "dropout": dropout},
    data=data
)
ckpt = torch.load(benign_model_path, map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)

# ---------------- 准备水印优化变量 ----------------
print(f"model layer keys: {ckpt.keys()}")
last_conv = model.convs[-1]
W_origin = last_conv.lin.weight.detach()  # shape: (C, D_hidden)

x_wm = data.x.clone()
preset_directions = torch.randn(n_classes, embedding_dim, device=device)
preset_directions /= torch.norm(preset_directions, dim=1, keepdim=True)

# ---------------- 采样每个类别的水印节点 ----------------
all_node_ids_list = []
x_sub_list = []
edge_index_sub_list = []
mapping_list = []
class_indices = []

for c in range(n_classes):
    idx_all = (data.y == c).nonzero(as_tuple=True)[0]
    node_ids = idx_all[torch.randperm(len(idx_all))[:nodes_per_class]]
    all_node_ids_list.append(node_ids)
    class_indices.extend([c] * node_ids.size(0))

    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=gnn_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    x_sub_list.append(data.x[subset])
    edge_index_sub_list.append(edge_index_sub)
    mapping_list.append(mapping)

# ---------------- 生成全局优化变量 ----------------
x_nodes_all = torch.cat([data.x[ids] for ids in all_node_ids_list], dim=0).clone().detach().to(device).requires_grad_(True)
class_indices = torch.tensor(class_indices, device=device)

# ---------------- 初始化 carriers ----------------
carriers = [preset_directions[c] for c in range(n_classes)]

# ---------------- 预计算 ft_orig ----------------
ft_orig_list = []
for c in range(n_classes):
    with torch.no_grad():
        ft_orig = model.encode(x_sub_list[c], edge_index_sub_list[c])[mapping_list[c]]
    ft_orig_list.append(ft_orig)

# ---------------- 打印迭代前信息 ----------------
print("\n=== Iteration 0 statistics (before any optimization) ===")
start_idx = 0
for c in range(n_classes):
    end_idx = start_idx + len(all_node_ids_list[c])
    delta_phi = torch.zeros_like(ft_orig_list[c])
    direction = carriers[c]

    cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
    alpha_std = torch.sum(delta_phi * direction, dim=1).std().item()
    ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
    x_l2_dist = torch.norm(x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]], dim=1).mean().item()
    x_max_amplitude = (x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]]).abs().max().item()

    print(f"Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
          f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
          f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f}")

    start_idx = end_idx

x_nodes_orig = x_nodes_all.detach().clone()

# ---------------- SGD 优化 ----------------
base_lr = lr
optimizer = optim.SGD([x_nodes_all], lr=lr, momentum=momentum)
g_ref = None
last_cos = 0.0
node_start_idx_list = np.cumsum([0] + [len(ids) for ids in all_node_ids_list])

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    loss_total = 0.0

    # 构造全图 patched 特征，只修改水印节点
    x_patched_full = data.x.clone()
    x_patched_full[torch.cat(all_node_ids_list)] = x_nodes_all

    # 每个类别计算 loss
    delta_phi_list_all = []
    start_idx = 0
    for c in range(n_classes):
        end_idx = node_start_idx_list[c+1]
        x_nodes_c = x_nodes_all[start_idx:end_idx]

        x_sub = x_sub_list[c]
        edge_index_sub = edge_index_sub_list[c]
        mapping = mapping_list[c]

        x_sub_patched = x_sub.clone()
        # 只更新水印节点
        x_sub_patched[mapping] = x_patched_full[all_node_ids_list[c]]

        ft_new = model.encode(x_sub_patched, edge_index_sub)[mapping]
        delta_phi = ft_new - ft_orig_list[c]
        delta_phi_list_all.append(delta_phi)

        direction = carriers[c]
        # loss_align = -torch.sum(delta_phi * direction)
        loss_align = - torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1).sum()
        loss_total += loss_align

        start_idx = end_idx

    loss_total.backward()
    # torch.nn.utils.clip_grad_norm_([x_nodes_all], max_norm=1.0)
    g = x_nodes_all.grad.norm().detach()
    if g_ref is None:
        g_ref = g.clone()

    ratio = (g / g_ref).clamp(0.05, 1.0)

    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * ratio.item()

    if ratio < 0.3 and isinstance(optimizer, optim.SGD):
        # 根据 ratio 动态设置 Adam lr，保持更新幅度合理
        adam_lr = max(max_adam_lr, base_lr * ratio)  # 保证不会太小
        optimizer = optim.Adam([x_nodes_all], lr=adam_lr)
        print(f"Switch to Adam with lr={adam_lr:.4f} at epoch {epoch}")
    optimizer.step()
    
    # ---------- L∞ 投影 ----------
    with torch.no_grad():
        x_nodes_all.copy_(torch.clamp(
            x_nodes_all,
            x_nodes_orig - init_clamp_max_perturb,
            x_nodes_orig + init_clamp_max_perturb
        ))
    
    # boundary-aware expansion
    step_size_ratio = (x_nodes_all - x_nodes_orig).abs().max() / init_clamp_max_perturb
    if step_size_ratio > 0.8:
        init_clamp_max_perturb *= 1.2
        init_clamp_max_perturb = min(init_clamp_max_perturb, perturb_amplitude)

    # cos-based annealing
    current_cos = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1).mean()
    if epoch % 20 == 0:
        if current_cos - last_cos < 1e-4:
            base_lr *= 0.7
        last_cos = current_cos

    # ---------------- 打印迭代信息 ----------------
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"\nEpoch {epoch+1:03d} | Total Loss: {loss_total.item():.6f}")
        start_idx = 0
        for c in range(n_classes):
            end_idx = node_start_idx_list[c+1]
            delta_phi = delta_phi_list_all[c]
            direction = carriers[c]

            cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
            alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
            alpha_std = torch.sum(delta_phi * direction, dim=1).std().item()
            ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
            x_l2_dist = torch.norm(x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]], dim=1).mean().item()
            x_max_amplitude = (x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]]).abs().max().item()
            cos_w = torch.nn.functional.cosine_similarity(direction.unsqueeze(0), W_origin[c].unsqueeze(0), dim=1).item()
            
            # print(f"cos_phi: {cos_phi}")
            print(f"Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
                  f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
                  f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f} | "
                  f"cos_w: {cos_w:.6f}")
            start_idx = end_idx

# ---------------- 保存 carriers 和水印节点 ----------------
start_idx = 0
mark_node_list = []
mark_strenth = 0.0
cosine_mean = 0.0
print("\n=== Final statistics after optimization ===")
for c in range(n_classes):
    n_nodes = all_node_ids_list[c].size(0)
    end_idx = start_idx + n_nodes
    x_nodes_c = x_nodes_all[start_idx:end_idx].detach()
    x_wm[all_node_ids_list[c]] = x_nodes_c
    mark_node_list.extend(all_node_ids_list[c].cpu().tolist())

    # ---------- 使用 x_nodes_orig 计算最大扰动 ----------
    delta_phi_final = delta_phi_list_all[c]
    direction = carriers[c]
    cos_phi = torch.nn.functional.cosine_similarity(delta_phi_final, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi_final * direction, dim=1).mean().item()
    alpha_std = torch.sum(delta_phi_final * direction, dim=1).std().item()
    ft_l2_dist = torch.norm(delta_phi_final, dim=1).mean().item()
    x_l2_dist = torch.norm(x_nodes_c - x_nodes_orig[start_idx:end_idx], dim=1).mean().item()
    x_max_amplitude = (x_nodes_c - x_nodes_orig[start_idx:end_idx]).abs().max().item()
    cos_w = torch.nn.functional.cosine_similarity(direction.unsqueeze(0), W_origin[c].unsqueeze(0), dim=1).item()
    
    print(f"Final Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
          f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
          f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f} | "
          f"cos_w: {cos_w:.6f}")
    
    mark_strenth += alpha_mean
    cosine_mean += cos_phi.mean().item()

    start_idx = end_idx
print(f"Overall mark strength: {(mark_strenth / n_classes):.6f}")
print(f"Overall cosine mean: {(cosine_mean / n_classes):.6f}")

# ---------------- 保存结果 ----------------
torch.save(torch.stack(carriers, dim=0).cpu(), carriers_save_path)
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

print(f"\nSaved watermarked dataset to {save_mark_dataset_path}")
print(f"Total marked nodes: {len(mark_node_list)}")
print("Watermarking process completed.")
