import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, LogSoftmax, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GCNConv, TAGConv
from torch_geometric.nn.norm import BatchNorm

def non_linearity(U, min, max):
  ''' U is the output of the GNN, BxNx3 (Qgen, V, angle)
      a and b Nx3 are the upper and lower limits for the three magnitudes '''
  a_batch = min.repeat(U.shape[0],1,1)
  b_batch = max.repeat(U.shape[0],1,1)

  gamma = a_batch + (b_batch-a_batch) / (1 + torch.exp(U))
  return gamma

class GNNUnsupervised(nn.Module):
    def __init__(self, dim, edge_index, Y_bus, num_layers, K,val_min,val_max, num_nodes,batch_norm=True):
        super(GNNUnsupervised, self).__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        self.batch_norm = batch_norm
        self.num_nodes = num_nodes
        self.dim = dim
        for layer in range(num_layers):
          self.convs.append(TAGConv(dim[layer],dim[layer+1],K[layer],bias=True))
          self.batchnorm.append(BatchNorm(num_nodes*dim[layer+1]))
        self.edge_index = edge_index
        self.edge_weights = torch.abs(Y_bus)
        self.relu = LeakyReLU()


    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        convs = self.convs
        relu = self.relu
        edge_index = self.edge_index
        edge_weights = self.edge_weights

        out = x
        for layer in range(num_layers-1):
          out = convs[layer](out,edge_index)
          if self.batch_norm:
             out = out.view(-1, self.num_nodes * self.dim[layer + 1])
             out = self.batchnorm[layer](out)
             out = out.view(-1, self.num_nodes, self.dim[layer + 1])
          out = relu(out)

        out = convs[-1](out,edge_index)
        out = non_linearity(out,self.val_min,self.val_max)
        return out