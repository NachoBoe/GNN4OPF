import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
from torch_geometric.data import Data
import pandapower as pp
import networkx as nx
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# sys.path.append(str(Path(__file__).parents[1]))
from src.arquitecturas import GNN_global, FCNN_global, GNN_Local
from src.Data_loader import load_net, load_data
from src.metric import NormalizedError
from src.train_eval import run_epoch, evaluate

import json 

import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser(description='Entrenar modelo')
parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
args = parser.parse_args()

# Load config file
cfg = OmegaConf.load(args.cfg)
outdir = Path(cfg.outdir) / cfg.data.red / cfg.model.model /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
weights_dir = outdir / 'weights'
weights_dir.mkdir(parents=True, exist_ok=True)

#Save cfg
OmegaConf.save(cfg, outdir / 'config.yaml')

# init tensorboard writer
writer = SummaryWriter(outdir)

# Set device
torch.manual_seed(cfg.training.seed)
device = cfg.training.device

# Set network
num_nodes, num_gens, edge_index, edge_weights, feature_mask, net = load_net(cfg.data.red,device)

# Set model
if cfg.model.model == 'GNN_global':
    model = GNN_global(cfg.model.layers,edge_index,edge_weights,len(cfg.model.layers)-1,cfg.model.K,num_nodes,num_gens, feature_mask, cfg.model.batch_norm).to(device)
elif cfg.model.model == 'FCNN_global':
    # Add in and out dimension
    cfg.model.layers[0] *= num_nodes
    # cfg.model.layers[-1] *= num_gens
    model = FCNN_global(cfg.model.layers,len(cfg.model.layers)-1,num_nodes, feature_mask, cfg.model.batch_norm).to(device)
elif cfg.model.model == 'GNN_local':
    model = GNN_Local(cfg.model.layers,edge_index,edge_weights,len(cfg.model.layers)-1,cfg.model.K,feature_mask,num_nodes,cfg.model.batch_norm).to(device)
elif cfg.model.model == 'FCNN_local':
    K = [0 for i in range(len(cfg.model.layers)-1)]
    model = GNN_Local(cfg.model.layers,edge_index,edge_weights,len(cfg.model.layers)-1,K,feature_mask,num_nodes,cfg.model.batch_norm).to(device)

# Load data
train_loader, val_loader, test_loader = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.normalize_Y,device)

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr,betas=cfg.training.betas,weight_decay=cfg.training.weight_decay)
criterion = nn.MSELoss()  # Change the loss function as needed

# Entrenamiento
best_acc = 1000
best_epoch = 0
last_train_metric_ploss = -1.
last_val_metric_ploss = -1.

for epoch in range(cfg.training.num_epochs):
    if epoch > cfg.training.initial_metric_epoch and epoch % cfg.training.metric_frec == 0:
        calculate_ploss_metric = True
    else:
        calculate_ploss_metric = False

    train_loss, train_metric, train_metric_ploss = run_epoch(model, train_loader, optimizer, criterion,calculate_ploss_metric, net, epoch,writer)
    val_loss, val_metric, val_metric_ploss  = evaluate(model, val_loader, criterion, calculate_ploss_metric, net, epoch,writer)
    if train_metric_ploss != None:
        last_train_metric_ploss = train_metric_ploss
    if val_metric_ploss != None:
        last_val_metric_ploss = val_metric_ploss

    print(f"Epoch {epoch+1}/{cfg.training.num_epochs}, Train Loss: {train_loss:.8f}, Train Metric: {train_metric:.4f}, Train Ploss: {last_train_metric_ploss:.4f},  Val Loss: {val_loss:.8f}, Val Metric: {val_metric:.4f},  Val Ploss: {last_val_metric_ploss:.4f}")
    # Save best model
    if val_metric < best_acc:
        best_acc = val_metric
        best_epoch = epoch
        torch.save(model.state_dict(), weights_dir / 'best_model.pt')
        ## save a json with the best values
        data = {
        'model_name': str(outdir),
        'val_loss': val_loss,
        'val_metric': val_metric,
        'val_p_loss': last_val_metric_ploss
        }

    # Serialize JSON data and write it to a file
        with open(outdir / 'best_model_info.json', 'w') as file:
            json.dump(data, file)
        
    # Early stopping
    if epoch - best_epoch > cfg.training.early_stopping:
        print(f"Early stopping at epoch {epoch}")
        break

# Evaluacion
test_loss, test_metric, test_ploss_metric = evaluate(model, test_loader, criterion,True,net,None,writer,test=True)
print(f"Test Loss: {test_loss:.8f}, Test Metric: {test_metric:.4f}")
print(f"Test Ploss: {test_ploss_metric:.4f}")

# log hyperparameters
writer.add_hparams(
    {
        'lr': cfg.training.lr,
        'beta0': cfg.training.betas[0],
        'beta1': cfg.training.betas[1],
        'weight_decay': cfg.training.weight_decay,
        'optimizer': 'Adam',
        'batch_size': cfg.training.batch_size,
    },
    {'hparam/accuracy': best_acc})
writer.close()
