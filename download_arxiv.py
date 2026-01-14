import torch
import random
import argparse
from collections import defaultdict

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import k_hop_subgraph
import warnings
warnings.filterwarnings("ignore")


def balanced_seed_sampling(y, train_mask, seeds_per_class, seed=42):
    """
    类别均衡采样 seed nodes（只从 train 节点中选）
    """
    torch.manual_seed(seed)
    random.seed(seed)

    y = y.view(-1)
    num_classes = int(y.max().item() + 1)

    class_to_nodes = defaultdict(list)

    for i in range(y.size(0)):
        if i in train_mask:
            class_to_nodes[int(y[i].item())].append(i)

    seed_nodes = []

    for c in range(num_classes):
        nodes = class_to_nodes[c]
        if len(nodes) == 0:
            continue

        if len(nodes) < seeds_per_class:
            print(f"[Warning] class {c} has only {len(nodes)} nodes")

        chosen = random.sample(nodes, min(seeds_per_class, len(nodes)))
        seed_nodes.extend(chosen)

    return torch.tensor(seed_nodes, dtype=torch.long)


def sample_balanced_subgraph(
    dataset,
    seeds_per_class=200,
    num_hops=2,
    seed=42
):
    """
    类别均衡 seed + k-hop 子图
    """
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    idx_train = split_idx["train"]
    idx_val = split_idx["valid"]
    idx_test = split_idx["test"]
    # --------------------------------------------------
    # 1. 均衡采样 seed nodes
    # --------------------------------------------------
    seed_nodes = balanced_seed_sampling(
        data.y,
        idx_train,
        seeds_per_class,
        seed
    )

    print(f"Number of seed nodes: {seed_nodes.numel()}")

    # --------------------------------------------------
    # 2. k-hop 子图
    # --------------------------------------------------
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=seed_nodes,
        num_hops=num_hops+1,
        edge_index=data.edge_index,
        relabel_nodes=True
    )

    # --------------------------------------------------
    # 3. 构建子图数据
    # --------------------------------------------------
    x_sub = data.x[subset]
    y_sub = data.y[subset].view(-1)
    
    num_nodes = data.num_nodes

    train_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask_full   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask_full  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask_full[idx_train] = True
    val_mask_full[idx_val]     = True
    test_mask_full[idx_test]   = True

    train_mask = train_mask_full[subset]
    val_mask = val_mask_full[subset]
    test_mask = test_mask_full[subset]

    sub_data = {
        "x": x_sub,
        "edge_index": edge_index_sub,
        "y": y_sub,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "orig_node_id": subset,        # 子图节点在原图中的 id
        "seed_node_id": seed_nodes     # watermark / probe 候选节点（原图 id）
    }

    return sub_data


def main(args):
    print("Loading ogbn-arxiv...")
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.root)

    data = dataset
    print(f"Original graph: {data.num_nodes} nodes")

    sub_data = sample_balanced_subgraph(
        dataset,
        seeds_per_class=args.seeds_per_class,
        num_hops=args.num_hops,
        seed=args.seed
    )

    print(
        f"Subgraph: {sub_data['x'].size(0)} nodes, "
        f"{sub_data['edge_index'].size(1)} edges"
    )

    torch.save(sub_data, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--seeds_per_class", type=int, default=50,
                        help="number of seed nodes per class")
    parser.add_argument("--num_hops", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/ogbn_arxiv_balanced_subgraph.pt")

    args = parser.parse_args()
    main(args)
