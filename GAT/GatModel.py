import torch.nn as nn
import torch.nn.functional as F
from GatLayer import GatModule


class GAT(nn.Module):
    """
    Graph Attention Network (GAT) for node classification.

    This class defines a 2-layer GAT model where the first layer learns attention weights
    on input features, and the second layer aggregates the features. The output is used
    for node classification tasks.

    Args:
        in_dim (int): Number of input features per node.
        hidden_dim (int): Number of hidden features for each attention head.
        out_dim (int): Number of output features (number of classes).
        heads (list): List of attention head counts for each layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, heads):
        super().__init__()

        # Define the first GAT layer with dropout and ELU activation
        self.layer1 = GatModule(
            in_dim,
            hidden_dim,
            heads[0],
            feat_drop=0.6,
            attn_drop=0.6,
            activation=F.elu,
        )

        # Define the second GAT layer
        self.layer2 = GatModule(
            hidden_dim * heads[0],
            out_dim,
            heads[1],
            feat_drop=0.6,
            attn_drop=0.6,
        )

    def forward(self, g, h):
        """
        Forward pass of the GAT model.

        Args:
            g (DGLGraph): Input graph.
            h (Tensor): Node features (N, in_dim).

        Returns:
            Tensor: Output node features (N, out_dim).
        """
        # Pass through the first GAT layer
        h = self.layer1(g, h)

        # Flatten the output for the second layer
        h = h.flatten(1)

        # Pass through the second GAT layer
        h = self.layer2(g, h)

        # Aggregate the results across heads (using mean)
        h = h.mean(1)

        return h


class GATppi(nn.Module):
    """
    Graph Attention Network (GAT) for the PPI (Protein-Protein Interaction) dataset.

    This model uses a 3-layer GAT architecture to predict node labels based on graph
    attention mechanisms, suitable for large-scale node classification tasks such as PPI.

    Args:
        in_dim (int): Number of input features per node.
        hidden_dim (int): Number of hidden features for each attention head.
        out_dim (int): Number of output features (number of classes).
        num_heads (list): List of attention head counts for each layer.
        residual (bool, optional): Whether to use residual connections. Defaults to False.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, residual=False):
        super().__init__()

        # First GAT layer
        self.layer1 = GatModule(
            in_dim,
            hidden_dim,
            num_heads[0],
            activation=F.elu,
            residual=residual,
        )

        # Second GAT layer
        self.layer2 = GatModule(
            hidden_dim * num_heads[0],
            hidden_dim,
            num_heads[1],
            activation=F.elu,
            residual=residual,
        )

        # Third GAT layer
        self.layer3 = GatModule(
            hidden_dim * num_heads[1],
            out_dim,
            num_heads[2],
            residual=residual,
        )

    def forward(self, g, h):
        """
        Forward pass of the GATppi model.

        Args:
            g (DGLGraph): Input graph.
            h (Tensor): Node features (N, in_dim).

        Returns:
            Tensor: Output node features (N, out_dim).
        """
        # Pass through the first GAT layer
        h = self.layer1(g, h)

        # Flatten the output for the second layer
        h = h.flatten(1)

        # Pass through the second GAT layer
        h = self.layer2(g, h)

        # Flatten again before the third layer
        h = h.flatten(1)

        # Pass through the third GAT layer
        h = self.layer3(g, h)

        # Aggregate the results (using mean) and return
        h = h.mean(1)

        return h
