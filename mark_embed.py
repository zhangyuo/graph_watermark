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


def project_linf_graph(x, x_orig, amplitude):
    """
    L-infinity projection for graph node features
    
    :param x: current optimized node features (Tensor)
    :param x_orig: original node features (Tensor)
    :param amplitude: maximum allowed perturbation per feature
    """
    delta = x - x_orig
    delta = torch.clamp(delta, -amplitude, amplitude)  # 限制每个维度
    return x_orig + delta


def build_x_sub_patched(x_sub_orig, mapping, x_nodes):
    x_sub = x_sub_orig.clone()
    for i, idx in enumerate(mapping):
        x_sub[idx] = x_nodes[i]
    return x_sub


def get_parser():
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument("--dump_path", type=str, default="", required=True)
    parser.add_argument("--carrier_path", type=str, default="", help="Direction in which to move features", required=True)
    parser.add_argument("--node_list", type=str, default=None, help="File that contains list of all nodes", required=True)
    parser.add_argument("--marking_network", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="", help="Graph dataset that contains the node embeddings", required=True)

    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--perturb_amplitude", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.5)
    parser.add_argument("--lambda_graph_l2", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    return parser


def main(params):
    logger = initialize_exp(params)

    params.node_list = [s.strip() for s in params.node_list.split(",")]
    print("Node list", params.node_list)

    # load graph dataset
    data = torch.load(params.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    print("Using device:", device)

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
    carrier_id = carrier_ids[0].item()  # all target nodes have the same label
    direction = torch.load(params.carrier_path).cuda()
    assert direction.dim() == 2
    direction = direction[carrier_id:carrier_id + 1]

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
        loss_ft = - torch.sum((ft - ft_orig) * direction)
        loss_ft_l2 = params.lambda_ft_l2 * torch.norm(ft - ft_orig, dim=1).sum()
        # loss_norm = 0
        # for i in range(len(x_nodes)):
        #     loss_norm += params.lambda_graph_l2 * torch.norm(x_nodes[i] - x_nodes_orig[i])**2
        loss_norm = sum(
            params.lambda_graph_l2 * torch.norm(x_nodes[i] - x_nodes_orig[i])**2
            for i in range(len(x_nodes))
        )
        loss = loss_ft + loss_norm + loss_ft_l2
        
        loss.backward()
        optimizer.step()

        # L-infinity projection
        with torch.no_grad():
            for i in range(len(x_nodes)):
                x_nodes[i].copy_(
                    project_linf_graph(x_nodes[i], x_nodes_orig[i], params.perturb_amplitude)
                )
        
        # Logging
        logs = {
            "keyword": "iteration",
            "loss": loss.item(),
            "loss_ft": loss_ft.item(),
            "loss_norm": loss_norm.item(),
            "loss_ft_l2": loss_ft_l2.item(),
        }
        if schedule is not None:
            logs["lr"] = schedule[iteration]
        logger.info("__log__:%s" % json.dumps(logs))
        
    with torch.no_grad():
        x_sub_final = build_x_sub_patched(x_sub, mapping, x_nodes)
        ft_new = model.encode(x_sub_final, edge_index_sub)[mapping]

    x_nodes_tensor = torch.stack(x_nodes, dim=0)
    logger.info("__log__:%s" % json.dumps({
        "keyword": "final",
        "ft_direction": torch.mv(ft_new - ft_orig, direction[0]).mean().item(),
        "ft_norm": torch.norm(ft_new - ft_orig, dim=1).mean().item(),
        "max_feat_change": (x_nodes_tensor - x_nodes_orig).abs().max().item()
    }))

    # Save original features (x_nodes_orig) and watermarked features (x_nodes)
    for i, node_id in enumerate(node_list):
        # Save original node features
        np.save(
            join(params.dump_path, f"node_{node_id}_orig.npy"),
            x_nodes_orig[i].detach().cpu().numpy().astype(np.float32)
        )

        # Save watermarked node features
        np.save(
            join(params.dump_path, f"node_{node_id}_wm.npy"),
            x_nodes_tensor[i].detach().cpu().numpy().astype(np.float32)
        )
    
    # save full watermarked x
    x_wm = data.x.clone()
    x_wm[node_list] = x_nodes_tensor.detach()

    torch.save({
        "x_orig": data.x.cpu(),
        "x_wm": x_wm.cpu(),
        "edge_index": data.edge_index.cpu(),
        "y": data.y.cpu(),
        "node_list": node_list.cpu(),
    }, join(params.dump_path, "graph_watermarked.pt"))


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
