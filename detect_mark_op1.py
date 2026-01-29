import numpy as np
import torch
import argparse
import time
import copy
import torch.nn.functional as F
from scipy.stats import combine_pvalues
from src.stats import cosine_pvalue
from model.GCN import MultiLayerGCN, build_gnn_model
from utility.utils import extract_node_embeddings
from torch_geometric.data import Data
from src.utils import initialize_exp
from mark_embed import analyze_delta_svd, compute_class_jacobian_svd_jvp
from torch_geometric.utils import k_hop_subgraph
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")

def load_classifier_weight(state_dict, key_candidates):
    """
    Load classifier weight from checkpoint state_dict.
    """
    for k in key_candidates:
        if k in state_dict:
            return state_dict[k].cpu().numpy()
    raise KeyError("Classifier weight not found in model state_dict")


if __name__ == "__main__":
    """
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client detect_mark.py --dataset_path mark_save/graph_watermarked_arxiv_dim128_layer2_seed42.pt --carrier_path mark_save/carriers_class40_dim128.pth --marking_model model_save/gcn_mark_arxiv_dim128_layer2_seed42.pth --benign_model model_save/gcn_benign_arxiv_dim128_layer2_seed42.pth --hidden_dim 128 --num_layers 2 --dropout 0.5
    """
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5
    seed_num = 42
    n_classes = 40

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=f"mark_save/graph_watermarked_arxiv_dim{hidden_dim}_layer{num_layers}_seed{seed_num}.pt", required=False)
    parser.add_argument("--carrier_path", type=str, default=f"mark_save/carriers_class{n_classes}_dim{hidden_dim}_uc.pth", help="Direction in which to move features", required=False)
    parser.add_argument("--marking_model", type=str, default=f"model_save/gcn_mark_arxiv_dim{hidden_dim}_layer{num_layers}_seed{seed_num}.pth", required=False)
    parser.add_argument("--benign_model", type=str, default=f"model_save/gcn_benign_arxiv_dim{hidden_dim}_layer{num_layers}_seed{seed_num}.pth", required=False)
    parser.add_argument("--hidden_dim", type=int, default=hidden_dim, required=False)
    parser.add_argument("--num_layers", type=int, default=num_layers, required=False)
    parser.add_argument("--dropout", type=float, default=dropout, required=False)

    parser.add_argument("--dump_path", type=str, default="logs", required=False)
    parser.add_argument("--exp_name", type=str, default="detect_mark", required=False)
    parser.add_argument("--exp_id", type=str, default=f"gcn_arxiv_dim{hidden_dim}_layer{num_layers}_seed{seed_num}", required=False)

    params = parser.parse_args()

    logger = initialize_exp(params)

    start_all = time.time()

    # --------------------------------------------------
    # Load graph data
    # --------------------------------------------------
    data = torch.load(params.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = data["x_orig"].clone().to(device)  # 原始特征
    if "node_list" in data and "x_wm" in data:
        node_list = data["node_list"].squeeze()
        x_wm_nodes = data["x_wm"][node_list].to(device)
        x[node_list] = x_wm_nodes  # 仅替换被标记节点的特征
    data = Data(
        x_orig=data["x_orig"],
        x_wm=x,
        edge_index=data["edge_index"],
        y=data["y"],
        train_mask=data["train_mask"],
        val_mask=data["val_mask"],
        test_mask=data["test_mask"],
        node_list=data["node_list"]
    )
    data = data.to(device)
    logger.info(f"Using device: {device}")
    num_train_nodes = data.train_mask.sum().item()
    logger.info(f"mark ratio: {len(data['node_list'])}/{num_train_nodes} = {len(data['node_list'])/num_train_nodes:.4f}")

    # --------------------------------------------------
    # Load watermark carrier
    # Shape: (num_classes, embedding_dim)
    # --------------------------------------------------
    carrier = torch.load(params.carrier_path)
    assert carrier.dim() == 2
    carrier = carrier.numpy()  # (C, D)
    logger.info(f"Carrier shape: {carrier.shape}")

    # --------------------------------------------------
    # Load marking (watermarked) model
    # --------------------------------------------------
    # params_1 = copy.deepcopy(params)
    # params_1.hidden_dim = 512
    marking_model = build_gnn_model(params, data)
    mark_ckpt = torch.load(params.marking_model)
    marking_model.load_state_dict(mark_ckpt)
    marking_model = marking_model.to(device).eval()

    # --------------------------------------------------
    # Load benign model
    # --------------------------------------------------
    benign_model = build_gnn_model(params, data)
    benign_ckpt = torch.load(params.benign_model)
    benign_model.load_state_dict(benign_ckpt)
    benign_model = benign_model.to(device).eval()

    # # ===== 在这里：compute Jacobian SVD =====
    # # get subgraph
    # subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
    #     node_idx=node_list,
    #     num_hops=params.num_layers + 1,
    #     edge_index=data.edge_index,
    #     relabel_nodes=True
    # )

    # x_sub = data.x_orig[subset]  # node features of the subgraph nodes

    # # Target nodes in subgraph
    # x_nodes_orig = x_sub[mapping]
    # x_nodes = [x.clone().detach().requires_grad_(True) for x in x_nodes_orig]
    # logger.info("Computing class Jacobian SVD for top-k directions...")
    # V_k = compute_class_jacobian_svd_jvp(
    #     model=benign_model,
    #     x_sub=x_sub,
    #     edge_index_sub=edge_index_sub,
    #     mapping=mapping,
    #     x_nodes=x_nodes,
    #     k=carrier.shape[1],
    #     sample_size=256
    # )
    # V_k = V_k.numpy()  # shape: (D, k)
    # logger.info(f"Computed top-{V_k.shape[1]} SVD directions for this class.")
    # V_k = torch.load(f"mark_save/svd_directions_k20_nodes5897.pth")  # shape: (D, k)
    # V_k = V_k.numpy()  # shape: (D, k)

    # --------------------------------------------------
    # Extract node embeddings (encoder output only)
    # IMPORTANT: same node set for both models
    # --------------------------------------------------
    with torch.no_grad():
        feat_mark = extract_node_embeddings(marking_model, data, use_wm=True)
        feat_benign = extract_node_embeddings(benign_model, data, use_wm=False)

    feat_mark = feat_mark.cpu().numpy()  # (N, D1)
    feat_benign = feat_benign.cpu().numpy()  # (N, D2)
    # feat_mark = feat_benign.copy()  # For testing purpose, use identical features
    # mark_ckpt = benign_ckpt.copy()  # For testing purpose, use identical weights

    logger.info(f"Marking feature shape: {feat_mark.shape}")
    logger.info(f"Benign feature shape : {feat_benign.shape}")

     # --------------------------------------------------
    node_list = data["node_list"].squeeze().cpu().numpy()
    labels = data["y"].cpu().numpy()
    marked_classes = np.unique(labels[node_list])
    logger.info(f"Marked classes: {marked_classes}")
    logger.info(f"Number of marked classes: {len(marked_classes)}")

    # --------------------------------------------------
    # Feature space alignment (least squares)
    # Solve: feat_mark @ X ≈ feat_benign, X shape: (D1, D2)
    # relative error	结论
    # < 0.1	两空间几乎线性等价
    # 0.1 – 0.3	可对齐
    # 0.3 – 0.6	结构差异明显
    # > 0.6	空间差异很大
    # --------------------------------------------------
    # X, residuals, rank, s = np.linalg.lstsq(feat_benign, feat_mark, rcond=None)
    # alignment_error = np.linalg.norm(feat_benign @ X - feat_mark) ** 2
    # logger.info(f"Number of nodes: {feat_mark.shape[0]}")
    # logger.info(f"Alignment residual norm: {alignment_error:.4e}")
    # rel_err = (
    #     np.linalg.norm(feat_benign @ X - feat_mark, 'fro') /
    #     np.linalg.norm(feat_mark, 'fro')
    # )
    # logger.info(f"Relative alignment error: {rel_err:.4f}")

    # class-wise least squares
    delta_feat = np.zeros_like(feat_mark)
    node_id_to_local = {nid: i for i, nid in enumerate(data["node_list"].squeeze().cpu().numpy())}
    X_c_dict = {}
    for c in marked_classes:
        # 所属类别 c 的节点
        nodes_c_global = np.array([nid for nid in node_list if labels[nid] == c])
        nodes_c = [node_id_to_local[nid] for nid in nodes_c_global]
        
        if len(nodes_c) <= 1:
            continue

        # 局部最小二乘
        X_c, *_ = np.linalg.lstsq(
            feat_benign[nodes_c],
            feat_mark[nodes_c],
            rcond=None
        )
        X_c_dict[c] = X_c

        # 局部对齐 embedding
        delta_feat[nodes_c] = feat_benign[nodes_c] @ X_c - feat_mark[nodes_c]

        # 可选 log：每类误差
        rel_err_c = np.linalg.norm(delta_feat[nodes_c], 'fro') / np.linalg.norm(feat_mark[nodes_c], 'fro')
        logger.info(f"Class {c}: relative alignment error = {rel_err_c:.4f}")




    # --------------------------------------------------
    # Weight-alignment-based watermark detection
    # --------------------------------------------------
    # Load mark classifier weight
    logger.info(f"model layer keys: {mark_ckpt.keys()}")
    last_conv = marking_model.convs[-1]
    W_mark = last_conv.lin.weight.detach().cpu().numpy()  # shape: (C, D_mark)

    # --------------------------------------------------
    # Project mark classifier weights into benign space
    # W_mark @ X
    # --------------------------------------------------
    # W_proj = np.dot(W_mark, X.T)

    W_proj = np.zeros_like(W_mark)
    for c in marked_classes:
        X_c = X_c_dict[c]
        W_proj[c] = np.dot(W_mark[c], X_c.T)  # shape: (D_benign,)

    # Normalize
    W_proj /= np.linalg.norm(W_proj, axis=1, keepdims=True)

    # --------------------------------------------------
    # Compute cosine scores
    # --------------------------------------------------
    scores = np.sum(W_proj * carrier, axis=1)
    scores = torch.nn.functional.cosine_similarity(benign_model.convs[-1].lin.weight.detach().cpu(), torch.load(params.carrier_path).cpu(), dim=1).numpy()
    # print("Cosine scores between carrier and classifier weights:", scores)

    # --------------------------------------------------
    # Statistical watermark detection
    # --------------------------------------------------
    p_vals_all = [cosine_pvalue(score, d=carrier.shape[1]) for score in scores]

    # Choose only marked classes for reporting
    weight_stats = {}
    for c in marked_classes:
        score = scores[c]
        p_c   = p_vals_all[c]
        z_c   = score / 1.0  # 单个向量 z-score 可以设为 score/std=1
        weight_stats[c] = {
            "score": score,
            "z": z_c,
            "num_vectors": 1,
            "p": p_c
        }
    # Combine p-values across marked classes
    pvals_weight_list = [weight_stats[c]["p"] for c in weight_stats]
    combined_weight_p = combine_pvalues(pvals_weight_list)[1]

    # logger.info per-class stats
    logger.info("====================================")
    logger.info(" Weight-alignment Watermark Detection")
    logger.info("====================================")
    for c, info in weight_stats.items():
        logger.info(
            f"Class {c:2d} | "
            f"score={info['score']:.4f} | "
            f"z={info['z']:.2f} | "
            f"vectors={info['num_vectors']} | "
            f"p={info['p']:.2e}"
        )
    logger.info("------------------------------------")
    logger.info(f"Mean score                  : {scores.mean():.4f}")
    logger.info(f"combined p-value            : {combined_weight_p:.2e}")
    logger.info(f"log10(combined p-value)     : {np.log10(combined_weight_p):.2f}")
    logger.info("------------------------------------")
    logger.info(f"Total time                  : {time.time() - start_all:.2f}s")
