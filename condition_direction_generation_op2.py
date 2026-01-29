# =============================================================
# Compute Cosine Similarity Upper Bound via Jacobian Image Space
# =============================================================
#
# This script numerically estimates the achievable upper bound of
#   max cos(Δφ, u)
# based on the geometric result:
#   max cos(Δφ, u) = || Proj_{Im(J)}(u) ||,  ||u||=1
#
# The Jacobian image space Im(J) is approximated by sampling
# reachable Δφ directions induced by small random perturbations
# on input node features.
#
# -------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from model.GCN import build_gnn_model
from utility.utils import set_seed

# ---------------- Configuration ----------------
seed = 42
set_seed(seed)

n_classes = 40
nodes_per_class = 100
embedding_dim = 512
gnn_layers = 2
dropout = 0.5
num_jacobian_samples = 64   # number of random perturbations
eps = 1e-3                 # perturbation magnitude
energy_ratio = 0.99        # retained SVD energy

# paths
dataset_path = "data/ogbn_arxiv_balanced_subgraph.pt"
benign_model_path = (
    f"model_save/gcn_benign_arxiv_dim{embedding_dim}_"
    f"layer{gnn_layers}_seed{seed}.pth"
)
carriers_path = f"mark_save/carriers_class{n_classes}_dim{embedding_dim}_uc.pth"

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Load Graph ----------------
loaded = torch.load(dataset_path)
data = Data(
    x=loaded["x"].to(device),
    edge_index=loaded["edge_index"].to(device),
    y=loaded["y"].to(device),
)

# ---------------- Load Model ----------------
model = build_gnn_model(
    params={"hidden_dim": embedding_dim, "num_layers": gnn_layers, "dropout": dropout},
    data=data
)
ckpt = torch.load(benign_model_path, map_location=device)
model.load_state_dict(ckpt)
model.eval().to(device)

# ---------------- Load target directions u ----------------
carriers = torch.load(carriers_path).to(device)  # [C, D]
carriers = F.normalize(carriers, dim=1)

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

@torch.no_grad()
def estimate_delta_phi_samples(
    model,
    x_sub,
    edge_index_sub,
    mapping,
    x_nodes_orig,
    num_samples=64,
    eps=1e-3,
):
    """
    Estimate reachable Δφ samples by random local perturbations.

    Returns:
        delta_phis: Tensor [K, D]
    """
    ft_orig = model.encode(x_sub, edge_index_sub)[mapping]
    delta_phis = []

    for _ in range(num_samples):
        noise = torch.randn_like(x_nodes_orig) * eps
        x_sub_perturbed = x_sub.clone()
        x_sub_perturbed[mapping] = x_nodes_orig + noise

        ft_new = model.encode(x_sub_perturbed, edge_index_sub)[mapping]
        delta_phi = (ft_new - ft_orig).mean(dim=0)  # aggregate nodes
        delta_phis.append(delta_phi)

    return torch.stack(delta_phis, dim=0)  # [K, D]


def compute_orthonormal_basis(delta_phis, energy_ratio=0.99):
    """
    Compute orthonormal basis of Im(J) approximation via SVD.

    Args:
        delta_phis: [K, D]
    Returns:
        B: [k, D] orthonormal basis (rows)
    """
    U, S, Vh = torch.linalg.svd(delta_phis, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    k = int((energy < energy_ratio).sum().item()) + 1
    return Vh[:k]


def cosine_upper_bound(u, B):
    """
    Compute ||Proj_{Im(J)}(u)||.
    u: [D], normalized
    B: [k, D], orthonormal basis
    """
    proj_coeff = B @ u
    return torch.norm(proj_coeff).item()

# -----------------------------------------------------------
# Main loop: per-class cosine upper bound
# -----------------------------------------------------------

upper_bounds = []

for c in range(n_classes):
    # ---- sample nodes of class c ----
    idx_all = (data.y == c).nonzero(as_tuple=True)[0]
    node_ids = idx_all[torch.randperm(len(idx_all))[:nodes_per_class]]

    subset, edge_index_sub, mapping, _ = k_hop_subgraph(
        node_idx=node_ids,
        num_hops=gnn_layers + 1,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    x_sub = data.x[subset]
    x_nodes_orig = data.x[node_ids]

    # ---- estimate Im(J) ----
    delta_phis = estimate_delta_phi_samples(
        model,
        x_sub,
        edge_index_sub,
        mapping,
        x_nodes_orig,
        num_samples=num_jacobian_samples,
        eps=eps,
    )

    # ---- orthonormal basis ----
    B = compute_orthonormal_basis(delta_phis, energy_ratio)

    # ---- cosine upper bound ----
    u = carriers[c]
    ub = cosine_upper_bound(u, B)
    upper_bounds.append(ub)

    print(f"Class {c:02d} | rank(Im(J)) ≈ {B.shape[0]:3d} | cosine upper bound ≈ {ub:.4f}")

# ---------------- Summary ----------------
upper_bounds = np.array(upper_bounds)
print("\n===== Summary =====")
print(f"Mean cosine upper bound: {upper_bounds.mean():.4f}")
print(f"Std  cosine upper bound: {upper_bounds.std():.4f}")
print(f"Min  cosine upper bound: {upper_bounds.min():.4f}")
print(f"Max  cosine upper bound: {upper_bounds.max():.4f}")

# End of file
