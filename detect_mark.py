import numpy as np
import torch
import argparse
import time

from scipy.stats import combine_pvalues
from src.stats import cosine_pvalue
from model.GCN import MultiLayerGCN, build_gnn_model
from utility.utils import extract_node_embeddings


def load_classifier_weight(state_dict, key_candidates):
    """
    Load classifier weight from checkpoint state_dict.
    """
    for k in key_candidates:
        if k in state_dict:
            return state_dict[k].cpu().numpy()
    raise KeyError("Classifier weight not found in model state_dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="", required=True)
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
    data = torch.load(params.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print("Using device:", device)

    # --------------------------------------------------
    # Load watermark carrier
    # Shape: (num_classes, embedding_dim)
    # --------------------------------------------------
    carrier = np.load(params.carrier_path)

    # --------------------------------------------------
    # Load marking (watermarked) model
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Extract node embeddings (encoder output only)
    # IMPORTANT: same node set for both models
    # --------------------------------------------------
    with torch.no_grad():
        feat_mark = extract_node_embeddings(marking_model, data, use_wm=True)
        feat_benign = extract_node_embeddings(benign_model, data, use_wm=False)

    feat_mark = feat_mark.cpu().numpy()  # (N, D1)
    feat_benign = feat_benign.cpu().numpy()  # (N, D2)

    print(f"Marking feature shape: {feat_mark.shape}")
    print(f"Benign feature shape : {feat_benign.shape}")

    # --------------------------------------------------
    # Feature space alignment (least squares)
    # Solve: feat_mark @ X ≈ feat_benign
    # --------------------------------------------------
    X, residuals, rank, s = np.linalg.lstsq(feat_mark, feat_benign, rcond=None)

    alignment_error = np.linalg.norm(feat_mark @ X - feat_benign) ** 2
    print(f"Alignment residual norm: {alignment_error:.4e}")

    # --------------------------------------------------
    # Load benign classifier weight
    # --------------------------------------------------
    key = 'fc.weight' # GCN classifier layer weight
    W_benign = benign_ckpt['model'][key].cpu().numpy()  # shape: (C, D_benign)

    # --------------------------------------------------
    # Project classifier weights into marking space
    # W_benign @ X^T
    # --------------------------------------------------
    W_proj = np.dot(W_benign, X.T)

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

    combined_p = combine_pvalues(p_vals)[1]

    print("====================================")
    print(" Graph Watermark Detection Result")
    print("====================================")
    print(f"Mean score        : {scores.mean():.4f}")
    print(f"log10(p-value)    : {np.log10(combined_p):.4f}")
    print(f"Epoch (benign)    : {benign_ckpt.get('epoch', -1)}")
    print(f"Total time        : {time.time() - start_all:.2f}s")
