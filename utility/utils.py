import random
import numpy as np
import argparse
import torch
from torch_geometric.utils import k_hop_subgraph


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Fix random seeds for full reproducibility.

    Args:
        seed (int): Random seed.
        deterministic (bool): Whether to enforce deterministic behavior (may reduce performance).
    """

    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_node_embeddings(model, data, use_wm=True):
    """
    Extract embeddings for watermarked nodes only, optionally using (l+1)-hop subgraph.

    Args:
        model: trained GNN model
        data: dict from graph_watermarked.pt
        use_wm: whether to use watermarked node features

    Returns:
        h_sel: Tensor of shape (K, D) for watermarked nodes
    """
    model.eval()

    # use watermarked or original feature embeddings
    x = data["x_wm"] if use_wm else data["x_orig"]
    edge_index = data["edge_index"]
    node_list = data["node_list"]

    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    node_list = node_list.to(device)

    # --------------------------------------------------
    # get embeddings for each watermarked node
    # --------------------------------------------------
    embeddings = []
    l = model.num_layers
    for node_id in node_list:
        node_id = node_id.item()
        subset, edge_index_sub, mapping, _ = k_hop_subgraph(
            node_idx=node_id,
            num_hops=l + 1,
            edge_index=edge_index,
            relabel_nodes=True
        )
        x_sub = x[subset]
        with torch.no_grad():
            h_sub = model.encode(x_sub, edge_index_sub)
        embeddings.append(h_sub[mapping[0]])  # mapping[0] denotes the target node

    h_sel = torch.stack(embeddings, dim=0)  # (K, D)
    return h_sel


def dict_to_argparser(params: dict):
    """
    Convert a parameter dict to argparse.ArgumentParser.
    
    Args:
        params (dict): parameter name -> default value
    
    Returns:
        argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()

    for key, value in params.items():
        arg_name = f"--{key}"

        # bool 单独处理
        if isinstance(value, bool):
            parser.add_argument(
                arg_name,
                action="store_true" if value is False else "store_false",
                help=f"(default: {value})"
            )
        else:
            parser.add_argument(
                arg_name,
                type=type(value),
                default=value,
                help=f"(default: {value})"
            )
    args = parser.parse_args()

    return args
