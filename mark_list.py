import torch
import random
from torch_geometric.data import Data
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")


def select_mark_nodes_any_class(
    data: Data,
    watermark_ratio: float,
    seed: int = 42,
    return_class_hist: bool = True,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Select watermark nodes from the training set according to a given ratio,
    without restricting to any specific class.

    Args:
        data: PyG Data object (must contain 'y' and 'train_mask').
        watermark_ratio: Fraction of training nodes to select as watermark nodes.
        seed: Random seed for reproducibility.
        return_class_hist: Whether to return class histogram of selected nodes.

    Returns:
        mark_list: List of selected node indices.
        class_hist (optional): {class_id: count} distribution of watermark nodes.
    """
    assert 0 < watermark_ratio <= 1.0, "watermark_ratio must be in (0, 1]"

    torch.manual_seed(seed)
    random.seed(seed)

    # --------------------------------------------------
    # 1. Get training nodes
    # --------------------------------------------------
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    num_train = train_nodes.numel()

    num_mark = max(1, int(num_train * watermark_ratio))
    if num_mark > num_train:
        raise ValueError("watermark_ratio too large.")

    # --------------------------------------------------
    # 2. Randomly sample watermark nodes
    # --------------------------------------------------
    perm = torch.randperm(num_train)[:num_mark]
    mark_nodes = train_nodes[perm].tolist()

    # --------------------------------------------------
    # 3. (Optional) class histogram for analysis
    # --------------------------------------------------
    class_hist = {}
    if return_class_hist:
        for n in mark_nodes:
            c = int(data.y[n].item())
            class_hist[c] = class_hist.get(c, 0) + 1

    return mark_nodes, class_hist


def select_mark_nodes(
    data: Data,
    watermark_ratio: float,
    target_class: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[int], int]:
    """
    Select watermark nodes from the training set according to a given ratio,
    optionally restricted to a single class.

    Args:
        data: PyG Data object (must contain 'y' and 'train_mask').
        watermark_ratio: Fraction of training nodes to select as watermark nodes (e.g., 0.05).
        target_class: Specific class to select nodes from. If None, selects the class 
                      with the most training nodes.
        seed: Random seed for reproducibility.

    Returns:
        mark_list: List of node indices selected for watermarking.
        target_class: The class used for selecting watermark nodes.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    # -------------------------
    # 1. Identify training nodes
    # -------------------------
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    num_train = train_nodes.size(0)

    num_mark = max(1, int(num_train * watermark_ratio))  # Ensure at least one node
    if num_mark == 0:
        raise ValueError("watermark_ratio is too small; no nodes would be selected.")

    # -------------------------
    # 2. Group training nodes by class
    # -------------------------
    class_to_nodes = {}
    for n in train_nodes.tolist():
        c = int(data.y[n].item())
        class_to_nodes.setdefault(c, []).append(n)

    # -------------------------
    # 3. Determine target class
    # -------------------------
    if target_class is None:
        # Default: choose the class with the largest number of training nodes
        target_class = max(class_to_nodes, key=lambda cls: len(class_to_nodes[cls]))

    candidate_nodes = class_to_nodes.get(target_class, [])

    if len(candidate_nodes) < num_mark:
        raise ValueError(
            f"Class {target_class} contains only {len(candidate_nodes)} training nodes, "
            f"but {num_mark} watermark nodes are required."
        )

    # -------------------------
    # 4. Sample watermark nodes
    # -------------------------
    mark_list = random.sample(candidate_nodes, num_mark)

    return mark_list, target_class


if __name__ == "__main__":
    """
    python mark_list.py
    """
    watermark = 20 # watermark ratio in percentage
    data_path = "data/ogbn_arxiv_balanced_subgraph.pt" # path to the graph data
    sub_data = torch.load(data_path)

    data = Data(
        x=sub_data["x"],
        edge_index=sub_data["edge_index"],
        y=sub_data["y"],
        train_mask=sub_data["train_mask"],
        val_mask=sub_data["val_mask"],
        test_mask=sub_data["test_mask"]
    )

    watermark_ratio = watermark / 100

    # mark_list, wm_class = select_mark_nodes(
    #     data,
    #     watermark_ratio=watermark_ratio,
    #     target_class=None,  # Automatically choose the class with the most training nodes
    #     seed=42
    # )
    # print(f"Watermark class: {wm_class}")

    mark_list, class_hist = select_mark_nodes_any_class(
        data,
        watermark_ratio=watermark_ratio,
        seed=42
    )
    print("Class distribution of watermark nodes:")
    for c, cnt in sorted(class_hist.items()):
        print(f"  class {c}: {cnt}")
    
    print(f"Number of watermark nodes: {len(mark_list)}")

    # -------------------------
    # Save watermark node list
    # -------------------------
    save_path = f"mark_save/mark_node_list_r{watermark}_num{len(mark_list)}_arxiv.pt"
    torch.save(mark_list, save_path)
    print(f"Watermark node list saved to: {save_path}")