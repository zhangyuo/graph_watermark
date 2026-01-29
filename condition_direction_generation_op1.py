import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from model.GCN import build_gnn_model
from utility.utils import set_seed
from mark_embed import build_x_sub_patched
import numpy as np
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
epochs = 1000
lr = 0.1
momentum = 0.9
perturb_amplitude = 100.0
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
last_conv = model.convs[-1]
W_origin = last_conv.lin.weight.detach()  # shape: (C, D_hidden)

x_wm = data.x.clone()
preset_directions = torch.ones(n_classes, embedding_dim, device=device)
# preset_directions = torch.randn(n_classes, embedding_dim, device=device)
preset_directions /= torch.norm(preset_directions, dim=1, keepdim=True)

# ---------------- 采样每个类别水印节点及其 k-hop 子图 ----------------
all_node_ids_list = []
x_sub_list = []
edge_index_sub_list = []
mapping_list = []
class_indices = []
subgraph_opt_indices = []

for c in range(n_classes):
    idx_all = (data.y == c).nonzero(as_tuple=True)[0]
    node_ids = idx_all[torch.randperm(len(idx_all))[:nodes_per_class]]
    all_node_ids_list.append(node_ids)
    class_indices.extend([c] * node_ids.size(0))

    # k-hop 子图
    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=gnn_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    x_sub_list.append(data.x[subset])
    edge_index_sub_list.append(edge_index_sub)
    mapping_list.append(mapping)

    # ---------- 计算子图内优化节点索引 ----------
    # 这里子图内的节点属于优化节点(all_opt_nodes)的索引
    # watermark 节点 + 1-hop 邻居
    neighbor_subset, _, _, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=1,
        edge_index=data.edge_index,
        relabel_nodes=False
    )
    all_opt_nodes_c = torch.tensor(list(set(node_ids.tolist()) | set(neighbor_subset.tolist())), device=device)
    mask = torch.tensor([i.item() in all_opt_nodes_c.tolist() for i in subset], device=device)
    subgraph_opt_indices.append(mask.nonzero(as_tuple=True)[0])

# ---------------- 全局优化节点集合 ----------------
all_opt_nodes = list(set([i.item() for sub in all_node_ids_list for i in sub] +
                         [i.item() for sub in subgraph_opt_indices for i in sub]))
all_opt_nodes_tensor = torch.tensor(all_opt_nodes, device=device)
x_opt_all = data.x[all_opt_nodes_tensor]
x_opt_all = x_opt_all.clone().detach().requires_grad_(True)
class_indices = torch.tensor(class_indices, device=device)

# ---------------- 初始化 carriers ----------------
carriers = [preset_directions[c] for c in range(n_classes)]

# ---------------- 预计算 ft_orig ----------------
ft_orig_list = []
for c in range(n_classes):
    with torch.no_grad():
        ft_orig = model.encode(x_sub_list[c], edge_index_sub_list[c])[mapping_list[c]]
    ft_orig_list.append(ft_orig)

# ---------------- 迭代前信息 ----------------
print("\n=== Iteration 0 statistics (before any optimization) ===")
start_idx = 0
for c in range(n_classes):
    delta_phi = torch.zeros_like(ft_orig_list[c])
    direction = carriers[c]
    cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
    start_idx += len(all_node_ids_list[c])
x_nodes_orig = x_opt_all.detach().clone()

# ---------------- SGD 优化 ----------------
optimizer = optim.SGD([x_opt_all], lr=lr, momentum=momentum)
node_start_idx_list = np.cumsum([0] + [len(ids) for ids in all_node_ids_list])

for epoch in range(epochs):
    optimizer.zero_grad()
    loss_total = 0.0

    # ---------- 构造全局 patched 特征 ----------
    x_patched_full = data.x.clone()
    x_patched_full[all_opt_nodes_tensor] = x_opt_all

    delta_phi_list_all = []
    start_idx = 0
    for c in range(n_classes):
        end_idx = node_start_idx_list[c+1]

        x_sub = x_sub_list[c].clone()
        edge_index_sub = edge_index_sub_list[c]
        mapping = mapping_list[c]

        # ---------- 替换子图内所有优化节点特征 ----------
        opt_indices = subgraph_opt_indices[c]
        x_sub[opt_indices] = x_patched_full[opt_indices]  # 直接使用 precomputed 索引

        ft_new = model.encode(x_sub, edge_index_sub)[mapping]
        delta_phi = ft_new - ft_orig_list[c]
        delta_phi_list_all.append(delta_phi)

        direction = carriers[c]
        cos_vals = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
        loss_total += -cos_vals.mean()

        start_idx = end_idx

    # ---------- 梯度归一化 ----------
    loss_total.backward()
    with torch.no_grad():
        grad = x_opt_all.grad
        grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
        x_opt_all.grad.copy_(grad)
    optimizer.step()

    # ---------- 打印 ----------
    if (epoch + 1) % 20 == 0 or epoch == 0:
        cos_means, alpha_means = [], []
        start_idx = 0
        for c in range(n_classes):
            end_idx = node_start_idx_list[c+1]
            delta_phi = delta_phi_list_all[c]
            direction = carriers[c]
            cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
            alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
            cos_means.append(cos_phi.mean().item())
            alpha_means.append(alpha_mean)
            start_idx = end_idx

        print(f"Epoch {epoch+1:03d} | Total Loss: {loss_total.item():.6f} | "
              f"Avg cos over classes: {np.mean(cos_means):.6f} | Avg alpha_mean: {np.mean(alpha_means):.6f}")

# ---------------- 保存 carriers 和水印节点 ----------------
mark_node_list, mark_strenth, cosine_mean = [], 0.0, 0.0
start_idx = 0
for c in range(n_classes):
    n_nodes = len(all_node_ids_list[c])
    end_idx = start_idx + n_nodes
    x_nodes_c = x_opt_all[start_idx:end_idx].detach()
    x_wm[all_node_ids_list[c]] = x_nodes_c
    mark_node_list.extend(all_node_ids_list[c].cpu().tolist())

    delta_phi_final = delta_phi_list_all[c]
    direction = carriers[c]
    cos_phi = torch.nn.functional.cosine_similarity(delta_phi_final, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi_final * direction, dim=1).mean().item()

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
