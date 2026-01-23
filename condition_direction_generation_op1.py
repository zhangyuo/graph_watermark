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
perturb_amplitude = 5.0
init_clamp_max_perturb = 1.0
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

# ---------------- 初始化水印方向 ----------------
x_wm = data.x.clone()
preset_directions = torch.randn(n_classes, embedding_dim, device=device)
preset_directions /= torch.norm(preset_directions, dim=1, keepdim=True)  # 单位化

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

# ---------------- 全局优化变量 ----------------
x_nodes_all = torch.cat([data.x[ids] for ids in all_node_ids_list], dim=0).clone().detach().to(device).requires_grad_(True)
class_indices = torch.tensor(class_indices, device=device)
carriers = [preset_directions[c] for c in range(n_classes)]

# ---------------- 预计算原始 embedding ----------------
ft_orig_list = []
for c in range(n_classes):
    with torch.no_grad():
        ft_orig = model.encode(x_sub_list[c], edge_index_sub_list[c])[mapping_list[c]]
    ft_orig_list.append(ft_orig)

x_nodes_orig = x_nodes_all.detach().clone()
node_start_idx_list = np.cumsum([0] + [len(ids) for ids in all_node_ids_list])

# ---------------- 优化器 ----------------
optimizer = optim.SGD([x_nodes_all], lr=lr, momentum=momentum)
g_ref = None

# ---------------- 优化循环 ----------------
for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    loss_total = 0.0
    delta_phi_list_all = []

    # 构造 patched 特征
    x_patched_full = data.x.clone()
    x_patched_full[torch.cat(all_node_ids_list)] = x_nodes_all

    # 每个类别计算 loss
    for c in range(n_classes):
        start_idx = node_start_idx_list[c]
        end_idx = node_start_idx_list[c+1]

        x_sub = x_sub_list[c]
        edge_index_sub = edge_index_sub_list[c]
        mapping = mapping_list[c]

        x_sub_patched = x_sub.clone()
        x_sub_patched[mapping] = x_patched_full[all_node_ids_list[c]]

        ft_new = model.encode(x_sub_patched, edge_index_sub)[mapping]
        delta_phi = ft_new - ft_orig_list[c]
        delta_phi_list_all.append(delta_phi)

        # ---------------- 只优化正向 cos ----------------
        direction = carriers[c]
        cos_vals = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
        cos_vals_clamped = torch.clamp(cos_vals, min=0.0)
        loss_align = - cos_vals_clamped.mean()
        loss_total += loss_align

    # ---------------- 反向传播 ----------------
    loss_total.backward()

    # 梯度归一化，防止节点梯度差异过大
    with torch.no_grad():
        grad = x_nodes_all.grad
        grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
        x_nodes_all.grad.copy_(grad)

    optimizer.step()

    # ---------------- 小幅度 L2 投影 ----------------
    with torch.no_grad():
        delta = x_nodes_all - x_nodes_orig
        delta_norm = delta.norm(dim=1, keepdim=True)
        scale = torch.clamp(delta_norm, max=perturb_amplitude) / (delta_norm + 1e-8)
        x_nodes_all.copy_(x_nodes_orig + delta * scale)

    # ---------------- 打印信息 ----------------
    if (epoch + 1) % 50 == 0 or epoch == 0:
        cos_means = []
        start_idx = 0
        for c in range(n_classes):
            end_idx = node_start_idx_list[c+1]
            delta_phi = delta_phi_list_all[c]
            direction = carriers[c]
            cos_vals = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
            cos_means.append(cos_vals.mean().item())
        print(f"Epoch {epoch+1:03d} | Avg cos over classes: {np.mean(cos_means):.6f} | Loss: {loss_total.item():.6f}")

# ---------------- 保存水印节点 ----------------
x_wm = data.x.clone()
mark_node_list = []
for c in range(n_classes):
    start_idx = node_start_idx_list[c]
    end_idx = node_start_idx_list[c+1]
    x_nodes_c = x_nodes_all[start_idx:end_idx].detach()
    x_wm[all_node_ids_list[c]] = x_nodes_c
    mark_node_list.extend(all_node_ids_list[c].cpu().tolist())

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
