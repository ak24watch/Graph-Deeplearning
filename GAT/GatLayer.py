import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
import dgl.function as fn


class GatModule(nn.Module):
    """
    A single layer of Graph Attention Network (GAT) implementation.

    This module computes the graph attention for each node, learning
    attention coefficients for each neighbor. It uses multi-head attention
    and supports residual connections.

    Args:
        in_feats (int): The number of input features per node.
        out_feats (int): The number of output features per node.
        num_heads (int): The number of attention heads to use in the multi-head attention.
        feat_drop (float, optional): Dropout rate for features. Defaults to 0.0.
        attn_drop (float, optional): Dropout rate for attention weights. Defaults to 0.0.
        negative_slope (float, optional): Slope of LeakyReLU activation. Defaults to 0.2.
        residual (bool, optional): Whether to use residual connections. Defaults to False.
        activation (nn.Module, optional): Activation function to apply to the output. Defaults to None.
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
    ):
        super().__init__()

        # Define a linear transformation for input features
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)

        # Attention mechanism parameters (left and right side)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        # Store relevant parameters
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        # Residual connection (optional)
        if residual:
            self.residual_fc = nn.Linear(in_feats, num_heads * out_feats)
        self.residual = residual

        # Initialize the model parameters
        self.resetParameters()

    def resetParameters(self):
        """
        Initialize the parameters of the model using Xavier Normalization.
        This includes the fully connected layer and the attention parameters.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        # Initialize residual connection (if used)
        if self.residual:
            nn.init.xavier_normal_(self.residual_fc.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        """
        Forward pass for the GAT layer.

        Args:
            graph (DGLGraph): The graph object containing the nodes and edges.
            feat (Tensor): Input feature tensor of shape (N, in_feats), where N is the number of nodes.
            edge_weight (Tensor, optional): Edge weights for each edge in the graph. Defaults to None.
            get_attention (bool, optional): If True, returns the attention coefficients. Defaults to False.

        Returns:
            Tensor: Output node features of shape (N, num_heads, out_feats).
        """
        with graph.local_scope():
            # Dropout applied to the input features
            h_src = h_dst = self.feat_drop(feat)

            # Compute the input features for attention mechanism (split into multiple heads)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self.num_heads, self.out_feats
            )

            # Attention scores (left and right side)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # Store computed features and attention values in the graph
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})

            # Apply edge function to compute attention scores
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            # Compute normalized attention scores using softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # Aggregating neighbors using the attention coefficients
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # Add residual connection (if enabled)
            if self.residual:
                res = self.residual_fc(h_dst).view(-1, self.num_heads, self.out_feats)
                rst = rst + res

            # Apply activation function (if any)
            if self.activation:
                rst = self.activation(rst)

            # Optionally return the attention coefficients
            if get_attention:
                return rst, graph.edata["a"]
            return rst
