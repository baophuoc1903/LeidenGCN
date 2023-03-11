import torch
from torch_geometric.nn import ClusterGCNConv
from torch_geometric.utils import dropout_edge

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ClusterGCNConv(args.in_channels, args.hidden_channels))
        for _ in range(args.num_layers - 2):
            self.convs.append(
                ClusterGCNConv(args.hidden_channels, args.hidden_channels))
        self.convs.append(
            ClusterGCNConv(args.hidden_channels, args.num_tasks))

        # Dropout
        self.dropout = torch.nn.Dropout(self.args.dropout)

        # Activation
        self.activation = torch.nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=0.1, training=self.training)
        for conv in self.convs[:-1]:

            x = conv(x, edge_index)
            # x = checkpoint(conv, x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x

