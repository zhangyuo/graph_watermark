import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from os.path import basename, join
from model.GCN import MultiLayerGCN, build_gnn_model
from src.data_augmentations import RandomResizedCropFlip, CenterCrop
from src.dataset import getImagenetTransform, NORMALIZE_IMAGENET
from src.datasets.folder import default_loader
from src.model import build_model
from src.utils import initialize_exp, bool_flag, get_optimizer, repeat_to
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import warnings
warnings.filterwarnings("ignore")


def project_linf_graph(x, x_orig, amplitude):
    """
    L-infinity projection for graph node features
    amplitude (L∞)	效果
    1e-1	embedding 爆炸 ❌
    1e-2	非常不稳定 ❌
    5e-3	勉强可控
    1e-3	✅ 推荐起点
    5e-4	稳定但 watermark 弱
    :param x: current optimized node features (Tensor)
    :param x_orig: original node features (Tensor)
    :param amplitude: maximum allowed perturbation per feature
    """
    delta = x - x_orig
    delta = torch.clamp(delta, -amplitude, amplitude)  # 限制每个维度
    return x_orig + delta


def project_l2_graph(x, x_orig, amplitude):
    """
    L2 projection for graph node features
    ||x - x_orig||_2 <= amplitude
    """
    delta = x - x_orig
    norm = torch.norm(delta)
    if norm > amplitude:
        delta = delta / (norm + 1e-12) * amplitude
    return x_orig + delta


def build_x_sub_patched(x_sub_orig, mapping, x_nodes):
    x_sub = x_sub_orig.clone()
    for i, idx in enumerate(mapping):
        x_sub[idx] = x_nodes[i]
    return x_sub


def analyze_delta_svd(delta: torch.Tensor, k: int = 20, logger=None):
    """
    对 delta 进行 SVD / PCA 分析，并返回主成分投影和奇异值统计

    :param delta: Tensor, shape [num_nodes, embedding_dim]
    :param k: 主成分维度，默认 20
    :param logger: 可选 logger，用于输出信息
    :return:
        delta_main: Tensor, [num_nodes, k] 投影到前 k 个主成分
        S: Tensor, 奇异值 [min(num_nodes, embedding_dim)]
        explained_ratio: Tensor, 每个奇异值贡献的方差比例
        cumulative_ratio: Tensor, 累积方差比例
        V_k: Tensor, [embedding_dim, k] 前 k 个主成分方向
    """
    # step 1: SVD
    U, S, V = torch.linalg.svd(delta, full_matrices=False)  # V: [D, D], S: [min(N,D)]
    
    # step 2: 取前 k 个主成分
    V_k = V[:, :k]                # [D, k]
    delta_main = delta @ V_k      # [num_nodes, k]

    # step 3: 奇异值贡献的方差
    explained_variance = S**2
    total_variance = explained_variance.sum()
    explained_ratio = explained_variance / total_variance

    # step 4: 累积方差
    cumulative_ratio = torch.cumsum(explained_ratio, dim=0)

    # step 5: 输出前 k 个主成分能量占比
    if logger is not None:
        logger.info(f"The energy of top-{k} parts: {cumulative_ratio[k-1].item():.4f}")

    return delta_main, V_k


def build_x_sub_patched_single(x_sub_orig, idx_in_sub, x_node):
    """
    仅替换 x_sub[idx_in_sub] 为 x_node
    """
    x_sub = x_sub_orig.clone()
    x_sub[idx_in_sub] = x_node
    return x_sub


def compute_class_jacobian_svd_jvp(model, x_sub, edge_index_sub, mapping, x_nodes, k=20, sample_size=128, n_probes=32):
    """
    快速近似 Jacobian SVD，用 JVP 替代完整 Jacobian
    """
    device = x_nodes[0].device

    if sample_size < len(x_nodes):
        idx = np.random.choice(len(x_nodes), sample_size, replace=False)
        x_nodes_sample = [x_nodes[i] for i in idx]
        mapping_sample = [mapping[i] for i in idx]
    else:
        x_nodes_sample = x_nodes
        mapping_sample = mapping

    Js_list = []

    for x_node, idx_in_sub in zip(x_nodes_sample, mapping_sample):
        x_node = x_node.detach().requires_grad_(True).to(device)

        def forward_fn(x):
            x_sub_patched = build_x_sub_patched_single(x_sub, idx_in_sub, x)
            return model.encode(x_sub_patched, edge_index_sub)[idx_in_sub]

        D = x_node.numel()
        # 生成随机 probe vectors
        probes = torch.randn(n_probes, D, device=device)

        # 对每个 probe 计算 Jv
        Jv_list = []
        for v in probes:
            _, Jv = torch.autograd.functional.jvp(forward_fn, x_node, v=v)
            Jv_list.append(Jv.flatten().detach().cpu())
        Js_list.append(torch.stack(Jv_list, dim=0))  # [n_probes, D]

    Js_stack = torch.cat(Js_list, dim=0)  # [num_nodes*n_probes, D]
    _, _, V = torch.linalg.svd(Js_stack, full_matrices=False)
    V_k = V[:, :k]
    return V_k


def get_parser():
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--mark_path", type=str, default="", required=True)
    parser.add_argument("--carrier_path", type=str, default="", help="Direction in which to move features", required=True)
    parser.add_argument("--marking_network", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, default="", help="Graph dataset that contains the node embeddings", required=True)
    parser.add_argument("--dataset_name", type=str, default="arxiv", help="Dataset name, e.g., arxiv, blogcatalog, etc.", required=True)
    parser.add_argument("--node_list_path", type=str, default="", required=True)

    parser.add_argument("--node_list", type=str, default=None, help="File that contains list of all nodes")
    parser.add_argument("--perturb_amplitude", type=int, default=10.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lambda_ft_l2", type=float, default=1.0)
    parser.add_argument("--lambda_graph_l2", type=float, default=1.0)
    # parser.add_argument("--mark_strength", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dump_path", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="mark_embed")
    parser.add_argument("--exp_id", type=str, default=f"benign_gcn_arxiv_dim{128}_layer{2}_seed{42}")
    parser.add_argument("--save_mark_dataset", type=str, default=f"graph_watermarked_arxiv_dim{128}_layer{2}_seed{42}.pt", help="Save path of watermarked dataset")

    return parser


def main(params):
    logger = initialize_exp(params)

    if params.node_list:
        params.node_list = [s.strip() for s in params.node_list.split(",")]
        logger.info(f"Node list: {params.node_list}")
    else:
        params.node_list = torch.load(params.node_list_path)
        logger.info(f"Loaded node list from: {params.node_list_path}")
        logger.info(f"Node list num: {len(params.node_list)}")

    # load graph dataset
    loaded = torch.load(params.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(loaded, Data):
        # Already a PyG Data object
        data = loaded
    else:
         # Assume it's a dict on arxiv subgraph and convert to Data
         data = Data(
            x=loaded["x"],
            edge_index=loaded["edge_index"],
            y=loaded["y"],
            train_mask=loaded.get("train_mask"),
            val_mask=loaded.get("val_mask"),
            test_mask=loaded.get("test_mask")
        )
    data = data.to(device)
    logger.info(f"Using device: {device}")

    # Build model / cuda
    model = build_gnn_model(params, data)
    model.to(device)

    ckpt = torch.load(params.marking_network)
    model.load_state_dict(ckpt)
    model = model.eval()

    # Loading carriers
    labels = data.y
    labels = labels.to(torch.long)
    node_list = torch.tensor(
        [int(i) for i in params.node_list],
        device=labels.device
    )
    carrier_ids = labels[node_list]
    # carrier_id = carrier_ids[0].item()  # all target nodes have the same label, becasue we adress only one class for each watermarking run
    directions = torch.load(params.carrier_path).cuda()
    assert directions.dim() == 2
    # direction = directions[carrier_id:carrier_id + 1]
    directions = directions[carrier_ids]  # (num_nodes, D)

    # get subgraph
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_list,
        num_hops=params.num_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    x_sub = data.x[subset]  # node features of the subgraph nodes

    # Target nodes in subgraph
    x_nodes_orig = x_sub[mapping]
    x_nodes = [x.clone().detach().requires_grad_(True) for x in x_nodes_orig]

    logger.info("Computing class Jacobian SVD for top-k directions...")
    V_k = compute_class_jacobian_svd_jvp(
        model=model,
        x_sub=x_sub,
        edge_index_sub=edge_index_sub,
        mapping=mapping,
        x_nodes=x_nodes,
        k=directions.shape[1],   # k = carrier 的维度
        sample_size=256           # 可调整，越大精度越高，显存消耗越大
    )
    logger.info(f"Computed top-{V_k.shape[1]} SVD directions for this class.")
    V_k = V_k.to(x_nodes_orig[0].device)  # 投回 GPU
    torch.save(V_k.cpu(), join(params.mark_path, f"svd_directions_k{directions.shape[1]}_nodes{len(node_list)}.pth"))

    optimizer, schedule = get_optimizer(x_nodes, params.optimizer)
    if schedule is not None:
        schedule = repeat_to(schedule, params.epochs)

    # Original embedding for watermarked nodes
    ft_orig = model.encode(x_sub, edge_index_sub)[mapping].detach()

    model = model.train()
    for iteration in range(params.epochs):
        optimizer.zero_grad()

        if schedule is not None:
            lr = schedule[iteration]
            logger.info("New learning rate for %f" % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Forward nodes
        x_sub_patched = build_x_sub_patched(x_sub, mapping, x_nodes)
        ft = model.encode(x_sub_patched, edge_index_sub)[mapping]

        # Losses
        # loss_ft = - torch.sum((ft - ft_orig) * directions)
        # loss_ft_l2 = params.lambda_ft_l2 * torch.norm(ft - ft_orig, dim=1).sum()
        # loss_norm = sum(
        #     params.lambda_graph_l2 * torch.norm(x_nodes[i] - x_nodes_orig[i])**2
        #     for i in range(len(x_nodes))
        # )

        # delta embedding
        delta = ft - ft_orig          # shape (num_nodes, D)
        # delta = analyze_delta_svd(delta, k=directions.shape[1], logger=logger)
        delta = delta @ V_k  # [num_nodes, k]

        # directions 已经是归一化向量
        proj = torch.sum(delta * directions, dim=1)     # 每个节点在 direction 上的投影 shape = (num_nodes,)

        # 计算 cosine similarity 强制 Δφ 对齐方向
        cos  = torch.nn.functional.cosine_similarity(delta, directions, dim=1)
        # delta_norm = delta / (delta.norm(dim=1, keepdim=True) + 1e-8)
        # loss_ft_cos = - 10000 * cos.mean() # 强制 Δφ 对齐方向
        # loss_ft_cos  = - 10000 * (delta_norm * directions).sum(dim=1).mean()
        # tau = 0.2   # 你希望达到的最小 cosine
        # loss_ft_cos = - 1000 * torch.mean(torch.relu(tau - cos))
        loss_ft_cos = - torch.tensor(0.0)

        # half-cone 正向对齐
        rho = 1.0  # 或 rho = 1 + np.tan(np.radians(angle))**2
        # proj_pos = torch.relu(proj)
        # loss_ft_cone = - rho * torch.sum(proj_pos ** 2)
        # loss_ft_cone = - rho * 100 * torch.sum(proj * torch.abs(proj)) # 强制 Δφ 在 direction 上有足够投影
        loss_ft_cone = - torch.sum(delta * directions)

        loss_ft = loss_ft_cone + loss_ft_cos
        loss_ft_l2 = params.lambda_ft_l2 * torch.norm(delta, dim=1).mean() # 控制 embedding 扰动幅度
        loss_x_l2 = torch.stack([
            torch.norm(x_nodes[i] - x_nodes_orig[i])**2
            for i in range(len(x_nodes))
        ]).mean() * params.lambda_graph_l2 # 控制节点特征扰动幅度

        loss = loss_ft + loss_ft_l2 + loss_x_l2
        
        loss.backward()
        optimizer.step()
        
        # Logging
        logs = {
            "iteration": iteration,
            "loss": loss.item(),
            "loss_ft_cone": loss_ft_cone.item(),
            "loss_ft_cos": loss_ft_cos.item(),
            "loss_ft_l2": loss_ft_l2.item(),
            "loss_x_l2": loss_x_l2.item(),
            "alpha_mean": torch.sum(delta * directions, dim=1).mean().item(),
            "alpha_std": torch.sum(delta * directions, dim=1).std().item(),
            "cosine_mean": torch.nn.functional.cosine_similarity(delta, directions, dim=1).mean().item(),
            "cosine_std": torch.nn.functional.cosine_similarity(delta, directions, dim=1).std().item(),
        }
        if schedule is not None:
            logs["lr"] = schedule[iteration]
        logger.info("__log__:%s" % json.dumps(logs))

        # L-infinity projection
        with torch.no_grad():
            for i in range(len(x_nodes)):
                x_nodes[i].copy_(
                    project_l2_graph(x_nodes[i], x_nodes_orig[i], params.perturb_amplitude)
                )
        
    with torch.no_grad():
        x_sub_final = build_x_sub_patched(x_sub, mapping, x_nodes)
        ft_new = model.encode(x_sub_final, edge_index_sub)[mapping]

    x_nodes_tensor = torch.stack(x_nodes, dim=0)
    delta = ft_new - ft_orig
    # delta = analyze_delta_svd(delta, k=directions.shape[1], logger=logger)
    delta = delta @ V_k  # [num_nodes, k]
    logger.info("__log__:%s" % json.dumps({
        "keyword": "final",
        "alpha_mean": torch.sum(delta * directions, dim=1).mean().item(),
        "alpha_std": torch.sum(delta * directions, dim=1).std().item(),
        "cosine_mean": torch.nn.functional.cosine_similarity(delta, directions, dim=1).mean().item(),
        "cosine_std": torch.nn.functional.cosine_similarity(delta, directions, dim=1).std().item(),
        "ft_l2": torch.norm(delta, dim=1).mean().item(),
        "x_max_amplitude": (x_nodes_tensor - x_nodes_orig).abs().max().item(),
        "rho": rho,
        "R": (rho * torch.sum((delta * directions)**2) - torch.sum(torch.norm(delta, dim=1)**2)).item()
    }))

    # # Save original features (x_nodes_orig) and watermarked features (x_nodes)
    # for i, node_id in enumerate(node_list):
    #     # Save original node features
    #     np.save(
    #         join(params.mark_path, f"node_{node_id}_orig.npy"),
    #         x_nodes_orig[i].detach().cpu().numpy().astype(np.float32)
    #     )

    #     # Save watermarked node features
    #     np.save(
    #         join(params.mark_path, f"node_{node_id}_wm.npy"),
    #         x_nodes_tensor[i].detach().cpu().numpy().astype(np.float32)
    #     )
    
    # save full watermarked x
    x_wm = data.x.clone()
    x_wm[node_list] = x_nodes_tensor.detach()

    if params.dataset_name == "bolgcatalog":
        torch.save({
            "x_orig": data.x.cpu(),
            "x_wm": x_wm.cpu(),
            "edge_index": data.edge_index.cpu(),
            "y": data.y.cpu(),
            "node_list": node_list.cpu(),
        }, join(params.mark_path, params.save_mark_dataset))
    elif params.dataset_name == "arxiv":
        torch.save({
            "x_orig": data.x.cpu(),
            "x_wm": x_wm.cpu(),
            "edge_index": data.edge_index.cpu(),
            "y": data.y.cpu(),
            "train_mask": data.train_mask.cpu(),
            "val_mask": data.val_mask.cpu(),
            "test_mask": data.test_mask.cpu(),
            "node_list": node_list.cpu(),
        }, join(params.mark_path, params.save_mark_dataset))


if __name__ == '__main__':
    """
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client mark_embed.py --carrier_path ./mark_save/carriers_class40_dim128.pth --epochs 300 --node_list_path ./mark_save/mark_node_list_r20_num5897_arxiv.pt --lambda_ft_l2 1.0 --lambda_graph_l2 1.0 --marking_network ./model_save/gcn_benign_arxiv_dim128_layer2_seed42.pth --mark_path ./mark_save --optimizer sgd,lr=0.01 --dataset ./data/ogbn_arxiv_balanced_subgraph.pt --dataset_name arxiv
    """

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
