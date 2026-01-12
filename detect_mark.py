import numpy as np
import torch
import argparse
import time
import copy
from scipy.stats import combine_pvalues
from src.stats import cosine_pvalue
from model.GCN import MultiLayerGCN, build_gnn_model
from utility.utils import extract_node_embeddings
from torch_geometric.data import Data


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

    # --------------------------------------------------
    # Load mark classifier weight
    # --------------------------------------------------
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
    p_vals = [
        cosine_pvalue(score, d=carrier.shape[1])
        for score in scores
    ]

    # choose p_vals according marked classes only
    node_list = sub_data["node_list"].squeeze().cpu().numpy()
    labels = sub_data["y"].cpu().numpy()
    marked_classes = np.unique(labels[node_list])
    p_vals_marked = [p_vals[c] for c in marked_classes]
    
    combined_p = combine_pvalues(p_vals_marked)[1]

    print("====================================")
    print(" Graph Watermark Detection Result")
    print("====================================")
    print(f"Single class score          : {scores[16]:.4f}")  # example for class 16
    print(f"Single class p-value        : {p_vals[16]:.4f}")
    print(f"log10(single class p-value) : {np.log10(p_vals[16]):.4f}")
    print("------------------------------------")
    print(f"Mean score                  : {scores.mean():.4f}")
    print(f"combined p-value            : {combined_p:.4f}")
    print(f"log10(combined p-value)     : {np.log10(combined_p):.4f}")
    print("------------------------------------")
    print(f"Epoch (benign)              : {benign_ckpt.get('epoch', -1)}")
    print(f"Total time                  : {time.time() - start_all:.2f}s")
