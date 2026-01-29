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
from src.stats import cosine_pvalue
from scipy.stats import combine_pvalues
warnings.filterwarnings("ignore")

# ---------------- 配置 ----------------
seed_num = 42
set_seed(seed_num)
seed_num = 42
model_type = "SurrogateGCN"
n_classes = 40
choose_mark_class = True
nodes_per_class = 1
embedding_dim = 128
gnn_layers = 2
dropout = 0.5
epochs = 100000
lr = 0.1 # [0.01, 0.1]
momentum = 0.9
perturb_amplitude = 1.0 # [0.1,5.0]
perturb_hop = 1
dataset_path = f"data/ogbn_arxiv_balanced_subgraph.pt"
benign_model_path = f"model_save_gcn_remove_relu/gcn_benign_arxiv_dim{embedding_dim}_layer{gnn_layers}_seed{seed_num}.pth"
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
    data=data,
    model_architecture=model_type
)
ckpt = torch.load(benign_model_path, map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)

# ---------------- 准备水印优化变量 ----------------
x_wm = data.x.clone()
# preset_directions = torch.ones(n_classes, embedding_dim, device=device)
# preset_directions = torch.abs(torch.randn(n_classes, embedding_dim, device=device))
preset_directions = torch.randn(n_classes, embedding_dim, device=device)
preset_directions /= torch.norm(preset_directions, dim=1, keepdim=True)
# preset_directions = torch.load(carriers_save_path).to(device)

# ---------------- 采样每个类别的水印节点 ----------------
all_node_ids_list = []
subset_list = []
x_sub_list = []
edge_index_sub_list = []
mapping_list = []
class_indices = []
subgraph_opt_indices = []
all_sub_nodes = []

expected_nodes = []
for c in range(n_classes):
    idx_all = (data.y == c).nonzero(as_tuple=True)[0]
    node_ids = idx_all[torch.randperm(len(idx_all))[:nodes_per_class]]
    node_ids = torch.tensor([33848], dtype=torch.long, device=idx_all.device) # 指定一个单一节点 33848
    # node_ids = torch.tensor([33848, 69293, 53469, 62654, 83144, 18370, 3201, 77383, 79592, 58113, 37326, 72412, 48989, 1478, 82774, 80830, 45060, 15679, 28835, 15419, 12613, 7380, 8316, 1376, 45668, 74076, 59939, 34314, 5177, 36055, 9857, 48386, 77980, 27309, 71775, 29520, 41876, 16133, 66542, 44801, 67859, 21588, 10786, 27886, 79985, 82144, 28821, 50418, 65181, 82135, 82872, 81165, 13747, 67985, 76983, 11126, 64072, 20295, 73167, 59489, 78618], dtype=torch.long, device=idx_all.device)
    all_node_ids_list.append(node_ids)
    class_indices.extend([c] * node_ids.size(0))

    # for i, k in enumerate(node_ids):
    #     subset, edge_index_sub, mapping, _ = k_hop_subgraph(
    #         node_idx=node_ids[i].unsqueeze(0),
    #         num_hops=gnn_layers + 1,
    #         edge_index=data.edge_index,
    #         relabel_nodes=True
    #     )
    #     if len(subset) > 1:
    #         # print(node_ids[i])
    #         # print(subset)
    #         expected_nodes.append(node_ids[i].item())
    # print(expected_nodes)

    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=gnn_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )
    subset_list.append(subset)
    x_sub_list.append(data.x[subset])
    edge_index_sub_list.append(edge_index_sub)
    mapping_list.append(mapping)

    # ---------- 计算子图内优化节点索引 ----------
    # 这里子图内的节点属于优化节点(all_opt_nodes)的索引
    # watermark 节点 + 1-hop 邻居
    neighbor_subset, _, _, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=perturb_hop,
        edge_index=data.edge_index,
        relabel_nodes=False
    )
    opt_nodes_global = torch.unique(torch.cat([node_ids, neighbor_subset]))
    all_sub_nodes.append(opt_nodes_global)
    # 映射到子图 index
    opt_indices_sub = torch.isin(subset, opt_nodes_global).nonzero(as_tuple=True)[0]
    subgraph_opt_indices.append(opt_indices_sub)

    if choose_mark_class:
        break

# ---------------- 生成全局优化变量 ----------------
all_opt_nodes = list(set([i.item() for sub in all_node_ids_list for i in sub] + [i.item() for sub in all_sub_nodes for i in sub]))
all_opt_nodes_tensor = torch.tensor(all_opt_nodes, device=device)
x_opt_all = data.x[all_opt_nodes_tensor]
x_opt_all = x_opt_all.clone().detach().to(device).requires_grad_(True)
class_indices = torch.tensor(class_indices, device=device)

# ---------------- 初始化 carriers ----------------
carriers = [preset_directions[c] for c in range(n_classes)]

# ---------------- 计算benign模型分类权重与预设方向的概率（假设检验） ----------------
print(f"model layer keys: {ckpt.keys()}")
last_conv = model
W_origin = last_conv.lin.weight.detach()  # shape: (C, D_hidden)
W_benign = W_origin.cpu().numpy()  # shape: (C, D_benign)
W_benign /= np.linalg.norm(W_benign, axis=1, keepdims=True)
ca = preset_directions.cpu().numpy()
scores = np.sum(W_benign * ca, axis=1)
p_vals_all = [cosine_pvalue(score, d=ca.shape[1]) for score in scores]
if choose_mark_class:
    print(f"p={p_vals_all[0]}")
    print(f"log10(p)={np.log10(p_vals_all[0])}")
else:
    print(f"log10(p)={np.log10(combine_pvalues(p_vals_all)[1])}")

# ---------------- 预计算 ft_orig ----------------
ft_orig_list = []
for c in range(n_classes):
    with torch.no_grad():
        ft_orig = model.encode(x_sub_list[c], edge_index_sub_list[c])[mapping_list[c]]
    ft_orig_list.append(ft_orig)
    if choose_mark_class:
        break

# ---------------- 打印迭代前信息 ----------------
print("\n=== Iteration 0 statistics (before any optimization) ===")
start_idx = 0
for c in range(n_classes):
    delta_phi = torch.zeros_like(ft_orig_list[c])
    direction = carriers[c]

    cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
    alpha_std = torch.sum(delta_phi * direction, dim=1).std().item()

    end_idx = start_idx + len(all_node_ids_list[c])

    ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
    # x_l2_dist = torch.norm(x_opt_all[start_idx:end_idx] - data.x[all_node_ids_list[c]], dim=1).mean().item()
    # x_max_amplitude = (x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]]).abs().max().item()

    # print(f"Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
    #       f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
    #       f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f}")

    start_idx = end_idx
    if choose_mark_class:
        break

x_nodes_orig = x_opt_all.detach().clone()


# ---------------- SGD 优化 ----------------
base_lr = lr
optimizer = optim.SGD([x_opt_all], lr=lr, momentum=momentum)
node_start_idx_list = np.cumsum([0] + [len(ids) for ids in all_node_ids_list])

for epoch in range(epochs):
    if epoch == 700:
        optimizer = optim.Adam([x_opt_all], lr=0.01)

    optimizer.zero_grad()
    loss_total = 0.0

    # 构造全图 patched 特征，修改目标节点和hop内节点
    x_patched_full = data.x.clone()
    x_patched_full[all_opt_nodes_tensor] = x_opt_all

    # 每个类别计算 loss
    delta_phi_list_all = []
    start_idx = 0
    for c in range(n_classes):
        end_idx = node_start_idx_list[c+1]

        x_sub = x_sub_list[c].clone()
        edge_index_sub = edge_index_sub_list[c]
        mapping = mapping_list[c]

        # ---------- 替换子图内所有优化节点特征 ----------
        subset = subset_list[c]
        opt_indices = subgraph_opt_indices[c]
        x_sub[opt_indices] = x_patched_full[subset[opt_indices]]  # 直接使用 precomputed 索引

        ft_new = model.encode(x_sub, edge_index_sub)[mapping]
        delta_phi = ft_new - ft_orig_list[c]
        delta_phi_list_all.append(delta_phi)

        direction = carriers[c]
        cos_vals = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
        # cos_vals = torch.sum(delta_phi * direction)
        loss_align = - cos_vals
        loss_total += loss_align

        start_idx = end_idx

        if choose_mark_class:
            break

    loss_total.backward()
    # 梯度归一化，防止节点梯度差异过大
    with torch.no_grad():
        grad = x_opt_all.grad
        grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
        x_opt_all.grad.copy_(grad)


    optimizer.step()
    
    # ---------- L∞ 投影 ----------
    # with torch.no_grad():
    #     x_nodes_all.copy_(torch.clamp(
    #         x_nodes_all,
    #         x_nodes_orig - perturb_amplitude,
    #         x_nodes_orig + perturb_amplitude
    #     ))
    # ---------------- 小幅度 L2 投影 ----------------
    # with torch.no_grad():
    #     delta = x_nodes_all - x_nodes_orig
    #     delta_norm = delta.norm(dim=1, keepdim=True)
    #     scale = torch.clamp(delta_norm, max=perturb_amplitude) / (delta_norm + 1e-8)
    #     x_nodes_all.copy_(x_nodes_orig + delta * scale)

    # ---------------- 打印迭代信息 ----------------

    if (epoch + 1) % 20 == 0 or epoch == 0:
        cos_means = []
        alpha_means = []
        start_idx = 0
        for c in range(n_classes):
            end_idx = node_start_idx_list[c+1]
            delta_phi = delta_phi_list_all[c]
            direction = carriers[c]

            cos_phi = torch.nn.functional.cosine_similarity(delta_phi, direction.unsqueeze(0), dim=1)
            alpha_mean = torch.sum(delta_phi * direction, dim=1).mean().item()
            alpha_std = torch.sum(delta_phi * direction, dim=1).std().item()

            ft_l2_dist = torch.norm(delta_phi, dim=1).mean().item()
            # x_l2_dist = torch.norm(x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]], dim=1).mean().item()
            # x_max_amplitude = (x_nodes_all[start_idx:end_idx] - data.x[all_node_ids_list[c]]).abs().max().item()
            cos_w = torch.nn.functional.cosine_similarity(direction.unsqueeze(0), W_origin[c].unsqueeze(0), dim=1).item()
            
            cos_means.append(cos_phi.mean().item())
            alpha_means.append(alpha_mean)
            start_idx = end_idx

            # print(f"cos_phi: {cos_phi}")
            # print(f"Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
            #       f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
            #       f"ft_l2_dist: {ft_l2_dist:.6f} | x_l2_dist: {x_l2_dist:.6f} | x_max_amplitude: {x_max_amplitude:.6f} | "
            #       f"cos_w: {cos_w:.6f}")

            if choose_mark_class:
                break
            
        print(f"Epoch {epoch+1:03d} | Total Loss: {loss_total.item():.6f} | "
              f"Avg cos over classes: {np.mean(cos_means):.6f} | Avg alpha_mean: {np.mean(alpha_means):.6f}")

# ---------------- 保存 carriers 和水印节点 ----------------
start_idx = 0
mark_node_list = []
mark_strenth = 0.0
cosine_mean = 0.0
print("\n=== Final statistics after optimization ===")
for c in range(n_classes):
    n_nodes = all_node_ids_list[c].size(0)
    end_idx = start_idx + n_nodes
    x_nodes_c = x_opt_all[start_idx:end_idx].detach()
    mark_node_list.extend(all_node_ids_list[c].cpu().tolist())

    # ---------- 使用 x_nodes_orig 计算最大扰动 ----------
    delta_phi_final = delta_phi_list_all[c]
    direction = carriers[c]
    cos_phi = torch.nn.functional.cosine_similarity(delta_phi_final, direction.unsqueeze(0), dim=1)
    alpha_mean = torch.sum(delta_phi_final * direction, dim=1).mean().item()
    alpha_std = torch.sum(delta_phi_final * direction, dim=1).std().item()

    ft_l2_dist = torch.norm(delta_phi_final, dim=1).mean().item()
    # x_l2_dist = torch.norm(x_nodes_c - x_nodes_orig[start_idx:end_idx], dim=1).mean().item()
    # x_max_amplitude = (x_nodes_c - x_nodes_orig[start_idx:end_idx]).abs().max().item()
    cos_w = torch.nn.functional.cosine_similarity(direction.unsqueeze(0), W_origin[c].unsqueeze(0), dim=1).item()
    
    print(f"Final Class {c} | alpha_mean: {alpha_mean:.6f} | alpha_std: {alpha_std:.6f} | "
          f"cosine_mean: {cos_phi.mean().item():.6f} | cosine_std: {cos_phi.std().item():.6f} | "
          f"ft_l2_dist: {ft_l2_dist:.6f} | "
          f"cos_w: {cos_w:.6f}")
    
    mark_strenth += alpha_mean
    cosine_mean += cos_phi.mean().item()

    start_idx = end_idx

    if choose_mark_class:
        break

x_wm[all_opt_nodes] = x_opt_all
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
