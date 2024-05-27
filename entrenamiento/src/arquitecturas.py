
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, GATConv
from torch_geometric.nn.norm import BatchNorm, InstanceNorm, GraphNorm

class GNN_global(nn.Module):
    def __init__(self, dim, edge_index,edge_weights, num_layers, K, num_nodes, num_gens, feature_mask, batch_norm=True):
        super(GNN_global, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.num_nodes = num_nodes
        self.num_gens = num_gens
        self.dim = dim
        self.edge_weights = edge_weights
        self.batch_norm = batch_norm
        self.feature_mask = feature_mask
        if self.batch_norm:
          self.batchnorm = torch.nn.ModuleList()
        for layer in range(num_layers-1):
          self.convs.append(TAGConv(dim[layer],dim[layer+1],K[layer],bias=True))
          if self.batch_norm:
            self.batchnorm.append(BatchNorm(num_nodes))
        
        self.fcnn = torch.nn.Linear(dim[-2]*num_nodes,dim[-1]*num_nodes)

        self.edge_index = edge_index
        self.relu = nn.LeakyReLU()

    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        convs = self.convs
        relu = self.relu
        fcnn = self.fcnn
        if self.batch_norm:
          batchnorm = self.batchnorm
        out = x
        for layer in range(num_layers-1):
          out = convs[layer](out,self.edge_index,self.edge_weights)
          if self.batch_norm:
            out = batchnorm[layer](out)
          out = relu(out)
        out = out.reshape(-1,self.dim[-2]*self.num_nodes)
        out = fcnn(out)
        out = out.reshape(-1, self.num_nodes, self.dim[-1])
        out *= self.feature_mask.unsqueeze(-1).repeat(1, self.dim[-1])
        # x = out.squeeze()
        
        return out

class FCNN_global(nn.Module):

    def __init__(self, dim, num_layers, num_nodes, feature_mask, batch_norm=True):
        super(FCNN_global, self).__init__()

        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        self.dim = dim
        self.batch_norm = batch_norm
        self.feature_mask = feature_mask
        if self.batch_norm:
          self.batchnorm = torch.nn.ModuleList()
        
        for layer in range(num_layers-1):
          self.linears.append(nn.Linear(dim[layer],dim[layer+1]))
          if self.batch_norm:
            self.batchnorm.append(BatchNorm(dim[layer+1]))
        self.relu = nn.LeakyReLU()

        self.linears.append(nn.Linear(dim[num_layers - 1], dim[-1] * num_nodes))



    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        linears = self.linears
        relu = self.relu
        if self.batch_norm:
          batchnorm = self.batchnorm

        out = x.reshape(-1,self.dim[0])

        for layer in range(num_layers-1):
          out = linears[layer](out)
          if self.batch_norm:
            out = batchnorm[layer](out)
          out = relu(out)
        
        out = linears[num_layers-1](out)
        out = out.reshape(-1, self.num_nodes, self.dim[-1])
        out *= self.feature_mask.unsqueeze(-1).repeat(1, self.dim[-1])
        return out


class GNN_Local(nn.Module):
    def __init__(self, dim, edge_index,edge_weights, num_layers, K, feature_mask,N, batch_norm=True):
        super(GNN_Local, self).__init__()
        self.batch_norm = batch_norm
        self.feature_mask = feature_mask
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.edge_weights = edge_weights
        self.dim = dim
        if self.batch_norm:
          self.batchnorm = torch.nn.ModuleList()
        for layer in range(num_layers):
          self.convs.append(TAGConv(dim[layer],dim[layer+1],K[layer],bias=True))
          if self.batch_norm:
            # self.batchnorm.append(GraphNorm(dim[layer+1]))
            self.batchnorm.append(BatchNorm(N))
        self.edge_index = edge_index
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        
        # Separate the feature mask from the node features
        feature_mask = self.feature_mask  # Extract the feature mask

        # Apply the GNN to the node features
        num_layers = self.num_layers
        convs = self.convs
        relu = self.relu
        if self.batch_norm:
          batchnorm = self.batchnorm
        out = x
        for layer in range(num_layers-1):
          out = convs[layer](out,self.edge_index, self.edge_weights)
          if self.batch_norm:
            # out = batchnorm[layer](out,batch_size=64)
            out = batchnorm[layer](out)
          out = relu(out)
        
        out = convs[-1](out,self.edge_index, self.edge_weights)
  

        # Apply the feature mask to the output features
        x = out * feature_mask.unsqueeze(-1).repeat(1, self.dim[-1])

        return x