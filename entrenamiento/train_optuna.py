import optuna
import torch
import os
import json
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import pandapower as pp
import networkx as nx
from torch.utils.tensorboard import SummaryWriter

from src.arquitecturas import GNNUnsupervised
from src.Data_loader import load_net, load_data
from src.train_eval import run_epoch, evaluate
from src.utils import get_Ybus, get_Yline, init_lamdas
from src.Loss import my_loss, get_max_min_values
from src.metric import feas_and_volt_metric

import warnings
warnings.filterwarnings('ignore')

def objective(trial):
    # Definir los espacios de búsqueda para c_1 y c_2
    c_1 = trial.suggest_loguniform('c_1', 0.01, 0.5)
    c_2 = trial.suggest_loguniform('c_2',0.1, 2)
    c_3 = 0
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    k = trial.suggest_int('k', 2, 5)
    layers =  trial.suggest_categorical('layers', [[4,32,32,5], [4,128,128,5], [4,512,512, 5], [4,2048,2048, 5], [4,32,32,32,5], [4,128,128,128,5], [4,512,512,512,5]])

    # Cargar configuración
    cfg = OmegaConf.load(args.cfg)
    outdir = Path(cfg.outdir) / cfg.data.red /  datetime.now().isoformat().split('.')[0][5:].replace('T', '_')
    weights_dir = outdir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración
    cfg.model.layers = layers
    cfg.model.K = [k] * (len(layers)-1)
    # cfg.data.normalize_X = norm_X
    cfg.training.dual_coefs = [c_1, c_2, c_3]
    cfg.training.lr = lr
    cfg.training.batch_size = batch_size
    OmegaConf.save(cfg, outdir / 'config.yaml')

    # Inicializar tensorboard
    writer = SummaryWriter(outdir)

    # Establecer dispositivo
    torch.manual_seed(cfg.training.seed)
    device = cfg.training.device

    # Cargar red
    edge_index, edge_weights, net = load_net(cfg.data.red, cfg.data.red_path, device)
    pp.runpp(net)
    Y_bus = get_Ybus(net, device)
    Y_line = get_Yline(net, device)
    max_ika = torch.Tensor(net.line["max_i_ka"]).to(device)

    # Cargar datos
    train_loader, val_loader, test_loader = load_data(cfg.data.data_path, cfg.training.batch_size, cfg.data.normalize_X, cfg.data.red, device)

    torch.autograd.set_detect_anomaly(True)

    dual_variables = init_lamdas(net, cfg.training.dual_coefs, device)
    min_vector, max_vector = get_max_min_values(net, device)

    num_layers = len(cfg.model.layers) - 1
    num_nodes = len(net.bus)
    model = GNNUnsupervised(cfg.model.layers, edge_index, Y_bus, num_layers, cfg.model.K, min_vector, max_vector, num_nodes, batch_norm=cfg.training.batch_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
    criterion = my_loss

    # Entrenar el modelo
    num_epochs = cfg.training.num_epochs
    best_loss = torch.inf
    best_epoch = 0
    for epoch in range(num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, Y_line, Y_bus, max_ika, dual_variables, epoch, writer)
        val_loss = evaluate(model, val_loader, criterion, Y_line, Y_bus, max_ika, dual_variables, epoch, writer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_epoch = epoch
            torch.save(best_model.state_dict(), weights_dir / 'best_model.pt')

        # Early stopping
        if epoch - best_epoch > cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Métricas de factibilidad y punto de ajuste de voltaje
    feas_metric, voltaje_set_metric, no_conv_count = feas_and_volt_metric(best_model, val_loader, net)
    
    data = {
        'model_name': str(outdir),
        'val_loss': val_loss,
        'feasibility_metric': feas_metric,
        'voltaje_setpoint_metric': voltaje_set_metric,
        'no_conv_count': no_conv_count
    }
    with open(outdir / 'best_model_info.json', 'w') as file:
        json.dump(data, file)

    writer.add_hparams(
        {
            'lr': cfg.training.lr,
            'beta0': cfg.training.betas[0],
            'beta1': cfg.training.betas[1],
            'weight_decay': cfg.training.weight_decay,
            'optimizer': 'Adam',
            'batch_size': cfg.training.batch_size,
            'c_1': c_1,
            'c_2': c_2
        },
        {'hparam/val_loss': best_loss})
    writer.close()

    return voltaje_set_metric #+ feas_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar modelo')
    parser.add_argument('--cfg', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=256)

    # print('Número de prueba: ', study.best_trial.number)
    # print('Mejores hiperparámetros: ', study.best_trial.params)
    # print('Mejor pérdida de validación: ', study.best_value)
