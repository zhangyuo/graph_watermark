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
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client detect_mark.py --dataset_path mark_save/graph_watermarked_arxiv.pt --carrier_path mark_save/carriers_class40_dim512.pth --marking_model model_save/gcn_mark_arxiv_dim512_layer2_seed42.pth --benign_model model_save/gcn_benign_arxiv_dim512_layer2_seed42.pth --hidden_dim 512 --num_layers 2 --dropout 0.5
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="", required=True)
    parser.add_argument("--carrier_path", type=str, default="", help="Direction in which to move features", required=True)
    parser.add_argument("--marking_model", type=str, required=True)
    parser.add_argument("--benign_model", type=str, required=True)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    params = parser.parse_args()

    start_all = time.time()

    # --------------------------------------------------
    # Load graph data
    # --------------------------------------------------
    sub_data = torch.load(params.dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = sub_data["x_orig"].clone().to(device)  # 原始特征
    if "node_list" in sub_data and "x_wm" in sub_data:
        node_list = sub_data["node_list"].squeeze()
        x_wm_nodes = sub_data["x_wm"][node_list].to(device)
        x[node_list] = x_wm_nodes  # 仅替换被标记节点的特征
    data = Data(
        x_orig=sub_data["x_orig"],
        x_wm=x,
        edge_index=sub_data["edge_index"],
        y=sub_data["y"],
        train_mask=sub_data["train_mask"],
        val_mask=sub_data["val_mask"],
        test_mask=sub_data["test_mask"],
        node_list=sub_data["node_list"]
    )
    data = data.to(device)
    print("Using device:", device)

    # --------------------------------------------------
    # Load watermark carrier
    # Shape: (num_classes, embedding_dim)
    # --------------------------------------------------
    carrier = torch.load(params.carrier_path)
    assert carrier.dim() == 2
    carrier = carrier.numpy()  # (C, D)
    print(f"Carrier shape: {carrier.shape}")

    # --------------------------------------------------
    # Load marking (watermarked) model
    # --------------------------------------------------
    params_1 = copy.deepcopy(params)
    params_1.hidden_dim = 512
    marking_model = build_gnn_model(params_1, data)
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

    print(f"Marking feature shape: {feat_mark.shape}")
    print(f"Benign feature shape : {feat_benign.shape}")

    # --------------------------------------------------
    # Feature space alignment (least squares)
    # Solve: feat_mark @ X ≈ feat_benign, X shape: (D1, D2)
    # --------------------------------------------------
    X, residuals, rank, s = np.linalg.lstsq(feat_mark, feat_benign, rcond=None)

    alignment_error = np.linalg.norm(feat_mark @ X - feat_benign) ** 2
    # --------------------------------------------------
    # relative error	结论
    # < 0.1	两空间几乎线性等价
    # 0.1 – 0.3	可对齐
    # 0.3 – 0.6	结构差异明显
    # > 0.6	空间差异很大
    # --------------------------------------------------
    print(f"Alignment residual norm: {alignment_error:.4e}")
    rel_err = (
        np.linalg.norm(feat_mark @ X - feat_benign, 'fro') /
        np.linalg.norm(feat_benign, 'fro')
    )
    print(f"Relative alignment error: {rel_err:.4f}")

    node_list = sub_data["node_list"].squeeze().cpu().numpy()
    labels = sub_data["y"].cpu().numpy()
    marked_classes = np.unique(labels[node_list])
    print(f"Marked classes: {marked_classes}")
    print(f"Number of marked classes: {len(marked_classes)}")

    # --------------------------------------------------
    # Embedding difference in aligned space
    # Δφ = φ_mark X - φ_benign
    # --------------------------------------------------
    delta_feat = feat_mark @ X - feat_benign   # shape: (N, D)
    
    # --------------------------------------------------
    # Projection-based watermark scores (node-level)
    # --------------------------------------------------
    proj_scores = {}
    proj_pvals  = {}

    D = carrier.shape[1]
    # 将 global id 映射为 delta_feat 的行索引
    node_id_to_local = {nid: i for i, nid in enumerate(node_list)}

    for c in marked_classes:
        # 所有属于类别 c 的 watermark 节点
        nodes_c_global = node_list[labels[node_list] == c]

        # 转换为 delta_feat 的局部索引
        nodes_c = [node_id_to_local[nid] for nid in nodes_c_global]

        if nodes_c == 0:
            continue

        d_c = carrier[c]              # (D,)
        d_c = d_c / np.linalg.norm(d_c)

        # 每个节点的投影值
        proj_vals = delta_feat[nodes_c] @ d_c   # shape: (num_nodes_c,)

        # 统计量：均值
        S_c = proj_vals.mean()
        std_c = proj_vals.std() + 1e-8

        # Z-score（近似正态）
        z_c = S_c / std_c

        proj_scores[c] = {
            "mean_proj": S_c,
            "std_proj": std_c,
            "z": z_c,
            "num_nodes": len(nodes_c)
        }

        # 单边 p-value（H0: mean <= 0）
        p_c = 1 - 0.5 * (1 + np.math.erf(z_c / np.sqrt(2)))
        proj_pvals[c] = p_c
    
    # --------------------------------------------------
    # Combine projection-based p-values
    # --------------------------------------------------
    proj_pvals_list = [proj_pvals[c] for c in proj_pvals]
    combined_proj_p = combine_pvalues(proj_pvals_list)[1]

    print("====================================")
    print(" Projection-based Watermark Detection")
    print("====================================")

    for c, info in proj_scores.items():
        print(
            f"Class {c:2d} | "
            f"mean_proj={info['mean_proj']:.4f} | "
            f"z={info['z']:.2f} | "
            f"nodes={info['num_nodes']} | "
            f"p={proj_pvals[c]:.2e}"
        )

    print("------------------------------------")
    print(f"Combined projection p-value : {combined_proj_p:.2e}")
    print(f"log10(p)                    : {np.log10(combined_proj_p):.2f}")


    # --------------------------------------------------
    # Model loss-based watermark detection
    # --------------------------------------------------
    loss_stats = {}
    marked_nodes = node_list  # global node id

    # 转换为 torch tensor
    labels_marked = torch.from_numpy(labels[marked_nodes]).to(device)

    with torch.no_grad():
        logits_mark = marking_model(data.x_wm, data.edge_index)   # 标记模型 logits
        logits_benign = benign_model(data.x_wm, data.edge_index)  # benign 模型 logits

    # 取水印节点对应 logits
    logits_mark_nodes = logits_mark[marked_nodes]
    logits_benign_nodes = logits_benign[marked_nodes]

    # 交叉熵 loss (per node)
    loss_mark_nodes = F.cross_entropy(logits_mark_nodes, labels_marked, reduction='none')
    loss_benign_nodes = F.cross_entropy(logits_benign_nodes, labels_marked, reduction='none')

    # Δloss = benign - marked
    delta_loss = (loss_benign_nodes - loss_mark_nodes).cpu().numpy()  # 正值 → 水印存在

    # 按类统计
    for c in marked_classes:
        nodes_c_mask = labels[marked_nodes] == c
        delta_c = delta_loss[nodes_c_mask]
        num_nodes_c = len(delta_c)
        if num_nodes_c == 0:
            continue

        mean_c = delta_c.mean()
        std_c  = delta_c.std() + 1e-8
        z_c    = mean_c / std_c
        p_c    = 1 - 0.5 * (1 + np.math.erf(z_c / np.sqrt(2)))  # 单边 p-value

        loss_stats[c] = {
            "mean_delta_loss": mean_c,
            "std_delta_loss": std_c,
            "z": z_c,
            "num_nodes": num_nodes_c,
            "p": p_c
        }

    # Combine p-values across classes
    pvals_loss_list = [loss_stats[c]["p"] for c in loss_stats]
    combined_loss_p = combine_pvalues(pvals_loss_list)[1]

    # 打印结果
    print("====================================")
    print(" Loss-based Watermark Detection")
    print("====================================")
    for c, info in loss_stats.items():
        print(
            f"Class {c:2d} | "
            f"mean_delta_loss={info['mean_delta_loss']:.4f} | "
            f"z={info['z']:.2f} | "
            f"nodes={info['num_nodes']} | "
            f"p={info['p']:.2e}"
        )
    print("------------------------------------")
    print(f"Combined loss-based p-value : {combined_loss_p:.2e}")
    print(f"log10(p)                    : {np.log10(combined_loss_p):.2f}")

    # --------------------------------------------------
    # Weight-alignment-based watermark detection
    # --------------------------------------------------
    # Load mark classifier weight
    key = 'classifier.weight' # GCN classifier layer weight
    W_mark = mark_ckpt[key].cpu().numpy()  # shape: (C, D_mark)

    # --------------------------------------------------
    # Project mark classifier weights into benign space
    # W_mark @ X
    # --------------------------------------------------
    W_proj = np.dot(W_mark, X)

    # Normalize
    W_proj /= np.linalg.norm(W_proj, axis=1, keepdims=True)
    carrier /= np.linalg.norm(carrier, axis=1, keepdims=True)

    # --------------------------------------------------
    # Compute cosine scores
    # --------------------------------------------------
    scores = np.sum(W_proj * carrier, axis=1)

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

    # Print per-class stats
    print("====================================")
    print(" Weight-alignment Watermark Detection")
    print("====================================")
    for c, info in weight_stats.items():
        print(
            f"Class {c:2d} | "
            f"score={info['score']:.4f} | "
            f"z={info['z']:.2f} | "
            f"vectors={info['num_vectors']} | "
            f"p={info['p']:.2e}"
        )
    print("------------------------------------")
    print(f"Mean score                  : {scores.mean():.4f}")
    print(f"combined p-value            : {combined_weight_p:.2e}")
    print(f"log10(combined p-value)     : {np.log10(combined_weight_p):.2f}")
    print("------------------------------------")
    print(f"Total time                  : {time.time() - start_all:.2f}s")
