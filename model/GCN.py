import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MultiLayerGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2, "at least 2-layer GCN"

        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # output GCN layer (linear)
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        # return penultimate embedding
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            # x_new = conv(x, edge_index)
            # x = x + x_new  # residual connection
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.convs[-1](x, edge_index)  # linear output
        return x
    
    def loss(self, logits, labels):
        """
        Classification loss (NLLLoss)

        logits: (N, C), raw outputs from forward(), that is: logits = model.forward(x, edge_index)
        labels: (N,)
        """
        log_prob = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_prob, labels)


def build_gnn_model(params, data):
    """
    Build a GNN model (MultiLayerGCN) for graph watermark detection.

    Args:
        params: dict or argparse.Namespace
            Must contain:
                - hidden_dim
                - num_layers
                - dropout
        data: torch_geometric.data.Data
            Used to infer:
                - num_node_features
                - num_classes

    Returns:
        model: torch.nn.Module
    """

    # -------- Infer dimensions from data --------
    num_features = data.num_node_features

    labels = data.y
    labels = labels.to(torch.long)
    num_classes = int(labels.max().item() + 1)

    # -------- Build model --------
    model = MultiLayerGCN(
        in_channels=num_features,
        hidden_channels=params["hidden_dim"]
        if isinstance(params, dict) else params.hidden_dim,
        out_channels=num_classes,
        num_layers=params["num_layers"]
        if isinstance(params, dict) else params.num_layers,
        dropout=params["dropout"]
        if isinstance(params, dict) else params.dropout,
    )

    return model


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
    val_acc   = (pred[data.val_mask]   == data.y[data.val_mask]).float().mean().item()
    test_acc  = (pred[data.test_mask]  == data.y[data.test_mask]).float().mean().item()
    return train_acc, val_acc, test_acc


def train_gnn_model(model, data, optimizer, epochs, model_save_path, logger=None):
    best_val_acc = 0
    best_train_acc = 0
    best_test_acc = 0

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)

        # Save the best test accuracy based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_train_acc = train_acc
            torch.save(model.state_dict(), model_save_path)

        if epoch % 10 == 0 or epoch == 1:
            if logger is not None:
                logger.info(
                    f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
                )
    print(f"Training finished! Best validation accuracy:{best_val_acc}, Train accuracy:{best_train_acc}, Test accuracy:{best_test_acc}")

    return best_val_acc, best_train_acc, best_test_acc
