import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class EGNNConv3D(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EGNNConv3D, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_pos = nn.Linear(3, out_channels)  # Linear transformation for 3D coordinates

    def forward(self, x, pos, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node features and positions.
        x = self.lin(x)
        pos = pos.cuda(0)
        pos_transformed = self.lin_pos(pos)

        # Step 3: Compute normalization

        row,col = edge_index

        # row=row[:30]
        # col = col[:30]
        deg = degree(row.cuda(0), x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)

        # deg_inv_sqrt = deg_inv_sqrt.cpu()
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Start propagating messages, including position information.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, pos=pos_transformed, norm=norm)

    # def message(self, x_j, pos_j, norm):
    #     # Combine node features and transformed positions with normalization.
    #     return norm.view(-1, 1) * (x_j + pos_j)

    def update(self, aggr_out):
        # Apply a nonlinear activation function to the aggregated output.
        return F.relu(aggr_out)