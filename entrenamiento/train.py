import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import os
import json
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
from src.arquitecturas import GNNUnsupervised
from src.Data_loader import load_net, load_data
from src.train_eval import run_epoch, evaluate
from src.utils import get_Ybus, get_Yline, init_lamdas
from src.Loss import my_loss, get_max_min_values

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    # Load config file
    cfg = OmegaConf.load(args.cfg)
    outdir = Path(cfg.outdir) / cfg.data.red /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
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
    edge_index, edge_weights, net = load_net(cfg.data.red,cfg.data.red_path,device)
    pp.runpp(net)
    Y_bus = get_Ybus(net,device)
    Y_line = get_Yline(net,device)
    max_ika = torch.Tensor(net.line["max_i_ka"]).to(device)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.red, device)

    torch.autograd.set_detect_anomaly(True)

    dual_variables = init_lamdas(net,device)
    min_vector, max_vector = get_max_min_values(net,device)

    num_layers = len(cfg.model.layers) - 1
    num_nodes = len(net.bus)
    model = GNNUnsupervised(cfg.model.layers, edge_index, Y_bus, num_layers, cfg.model.K, min_vector, max_vector, num_nodes,batch_norm=cfg.training.batch_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
    criterion = my_loss  # Change the loss function as needed

    # Train the model
    num_epochs = cfg.training.num_epochs
    best_loss = 1000
    best_epoch = 0
    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, Y_line, Y_bus, max_ika,dual_variables, epoch, writer)
        val_loss = evaluate(model, val_loader, criterion, Y_line, Y_bus, max_ika, dual_variables, epoch, writer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), weights_dir / 'best_model.pt')
            ## save a json with the best values
            data = {
            'model_name': str(outdir),
            'val_loss': val_loss,
            }
            with open(outdir / 'best_model_info.json', 'w') as file:
                json.dump(data, file)

        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Evaluate the model on the test set
    test_loss = evaluate(model, test_loader, criterion, Y_line, Y_bus, max_ika, dual_variables, epoch, writer,test=True)
    print(f"Test Loss: {test_loss:.4f}")

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
