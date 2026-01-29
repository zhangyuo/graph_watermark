import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv


class SurrogateGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # linear classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        """
        penultimate embedding: 多层 concat
        """
        hs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            # h = F.relu(h)
            # h = F.leaky_relu(h, negative_slope=0.01)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        # h_cat = torch.cat(hs, dim=1)  # multi-layer concat

        return h

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)


class SurrogateLinear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()

        # input layer
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))

        # linear classifier
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def encode(self, x, edge_index=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if len(self.layers) == i + 1:
                pass
            else:
                # h = F.relu(h)
                pass
            # h = F.leaky_relu(h, negative_slope=1)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index=None):
        h = self.encode(x)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)


class SurrogateX(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Y = X
        self.encoder = torch.nn.Identity()

        # classifier
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def encode(self, x, edge_index=None):
        h = self.encoder(x)      # Y = X
        # h = F.relu(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index=None):
        h = self.encode(x)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)


class SurrogateGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # linear classifier (多层 concat)
        self.lin = torch.nn.Linear(hidden_channels * num_layers, out_channels)

    def encode(self, x, edge_index):
        """
        penultimate embedding: 多层 concat
        """
        hs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h_cat = torch.cat(hs, dim=1)  # multi-layer concat
        return h_cat

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)
    

class SurrogateGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        # GIN 需要 MLP 作为每层的更新函数
        for layer in range(num_layers):
            if layer == 0:
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
            else:
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
            self.convs.append(GINConv(mlp))

        # linear classifier，支持多层 concat embedding
        self.lin = torch.nn.Linear(hidden_channels * num_layers, out_channels)

    def encode(self, x, edge_index, batch=None):
        """
        penultimate embedding: 多层 concat
        """
        hs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h_cat = torch.cat(hs, dim=1)  # multi-layer concat
        return h_cat

    def forward(self, x, edge_index, batch=None):
        """
        前向传播，输出分类 logits
        """
        h = self.encode(x, edge_index, batch)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)


class StandardGCN(torch.nn.Module):
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.convs[-1](x, edge_index)  # linear output
        return F.log_softmax(x, dim=1)
    
    def loss(self, preds, labels):
        """
        Classification loss (NLLLoss)

        logits: (N,)
        labels: (N,)
        """
        return F.nll_loss(preds, labels)


class SurrogateGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.heads = heads

        # input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))

        # linear classifier
        self.lin = torch.nn.Linear(hidden_channels * heads * num_layers, out_channels)

    def encode(self, x, edge_index):
        """
        penultimate embedding: 多层 concat
        """
        hs = []
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)  # GAT常用ELU
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h_cat = torch.cat(hs, dim=1)  # multi-layer concat
        return h_cat

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        logits = self.lin(h)
        return F.log_softmax(logits, dim=1)

    def loss(self, preds, labels):
        return F.nll_loss(preds, labels)


def build_gnn_model(params, data, model_architecture=None):
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
    if model_architecture is None:
        model = StandardGCN(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateGCN":
        model = SurrogateGCN(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateGraphSAGE":
        model = SurrogateGraphSAGE(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateGIN":
        model = SurrogateGIN(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateGAT":
        model = SurrogateGAT(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateLinear":
        model = SurrogateLinear(
            in_channels=num_features,
            hidden_channels=params["hidden_dim"]
            if isinstance(params, dict) else params.hidden_dim,
            out_channels=num_classes,
            num_layers=params["num_layers"]
            if isinstance(params, dict) else params.num_layers,
            dropout=params["dropout"]
            if isinstance(params, dict) else params.dropout,
        )
    elif model_architecture == "SurrogateX":
        model = SurrogateX(
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
