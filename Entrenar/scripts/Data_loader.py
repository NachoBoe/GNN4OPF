import pandapower as pp
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_net(red,red_path,device="cuda"):
    if red == '30':
        net = pp.networks.case30()
        z_trafos = net.trafo[['hv_bus', 'lv_bus']].to_numpy().astype(np.int32)
    elif red == '118':
        net = pp.networks.case118()
        z_trafos =  np.array([
            [0, 0.0267],
            [0, 0.0382],
            [0, 0.0388],
            [0, 0.0375],
            [0, 0.0386],
            [0, 0.0268],
            [0, 0.0370],
            [0.0013, 0.016],
            [0, 0.0370],
            [0.0017, 0.0202],
            [0, 0.0370],
            [0.0282, 0.2074],
            [0.0003, 0.00405]
        ])
    elif red == 'uru':
        net = pp.from_pickle(red_path)
        # r = net.trafo['vkr_percent'].to_numpy() / 100 * net.sn_mva / net.trafo['sn_mva'].to_numpy()
        r = net.trafo['vkr_percent'].to_numpy() / 100 * 100 / net.trafo['sn_mva'].to_numpy()
        # z = net.trafo['vkr_percent'].to_numpy() / 100 * net.sn_mva / net.trafo['sn_mva'].to_numpy()
        z = net.trafo['vk_percent'].to_numpy() / 100 * 100 / net.trafo['sn_mva'].to_numpy()
        x = np.sqrt(z**2 - r**2)
        z_trafos = np.stack((r,x),axis=1)

    num_nodes = len(net.bus)
    num_gens = len(net.gen) + len(net.ext_grid)
    
    # Armar matirz de adyacencia
    lineas = net.line[['from_bus', 'to_bus']].to_numpy().astype(np.int32)
    trafos = net.trafo[['hv_bus', 'lv_bus']].to_numpy().astype(np.int32)
    edge_index = np.append(lineas, trafos,axis=0)
    # edge_index = torch.Tensor(edge_index).t().type(torch.int64).to(device)

    # Esto se agrego para que ande en la de uru, anda tb en ieee
    indices_from = []
    indices_to = []
    for item in edge_index[:,0]:
        # Encuentra todos los índices en lista_principal para el item actual de otra_lista
        indices_from.extend([index for index, value in enumerate(net.bus.reset_index()['index'].to_list()) if value == item])
    for item in edge_index[:,1]:
        indices_to.extend([index for index, value in enumerate(net.bus.reset_index()['index'].to_list()) if value == item])
    indices = np.array([indices_from, indices_to])
    edge_index = torch.Tensor(indices).type(torch.int64).to(device)

    edge_index_T = edge_index.clone()
    edge_index_T[1, :] = edge_index[0, :]
    edge_index_T[0, :] = edge_index[1, :]
    edge_index = torch.cat((edge_index, edge_index_T), dim=1)

    # Armar matriz de pesos
    k = 10
    z_lineas =  net.line[['r_ohm_per_km', 'x_ohm_per_km']].to_numpy() * 100 / np.expand_dims((net.bus['vn_kv'][net.line['from_bus'].to_numpy()].to_numpy())**2,axis=1)
    edge_weights = np.append(z_lineas, z_trafos,axis=0)
    edge_weights = torch.Tensor(np.e**(-k*(edge_weights[:,0]**2 + edge_weights[:,1]**2))).to(device)
    edge_weights = torch.cat((edge_weights, edge_weights), dim=0)

    # Armar mascara de generadores para la salida

    # idx_gen = net.gen["bus"].to_numpy()
    # idx_grid = net.ext_grid["bus"].to_numpy()
    # idx_gens = np.append(idx_gen,idx_grid,axis=0)

    # Esto se agrego para que ande en la red de uru, pero tb anda en ieee
    idx_bus = net.bus.reset_index()['index'].to_list()
    idx_gen_bus = net.gen.bus.to_list()
    idx_grid_bus = net.ext_grid.bus.to_list()
    idx_gen = [i for i, num in enumerate(idx_bus) if num in idx_gen_bus]
    idx_grid = [i for i, num in enumerate(idx_bus) if num in idx_grid_bus]
    idx_gens = idx_gen + idx_grid

    feature_mask = np.zeros(len(net.bus.index), dtype=int)
    feature_mask[idx_gens] = 1
    feature_mask = torch.Tensor(feature_mask).type(torch.int32).to(device)

    print("edge_index",edge_index)
    print("edge_weights",len(edge_weights))

    return num_nodes, num_gens, edge_index, edge_weights, feature_mask, net




def load_data(data_path, batch_size, normalize_X, normalize_Y, device):
    
    # Levantar los datos
    X = np.load(os.path.join(data_path, 'input.npy'))
    y = np.load(os.path.join(data_path, 'vm_pu_opt.npy'))


    # Separar en X e Y
    X_tensor = torch.Tensor(X).to(device)
    y_tensor = torch.Tensor(y[:,:,0:1]).to(device)
    print("X_tensor", X_tensor.shape)
    print("y_tensor", y_tensor.shape)

    # dataset = TensorDataset(X_tensor, y_tensor)
    X_train,X_val,y_train,y_val = train_test_split(X_tensor,y_tensor,test_size=0.2,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.1,random_state=42)

    # Normalizar X
    if normalize_X:
        mean = torch.mean(X_tensor,0)

        std = torch.std(X_tensor,0)
        std[std == 0.] = float('inf')
        
        X_train  = (X_train - mean) / std
        X_val  = (X_val - mean) / std
        X_test  = (X_test - mean) / std

    # Normalizar y
    if normalize_Y:
        mean_y = torch.mean(y_train,0)

        std_y = torch.std(y_train,0)
        std_y[std_y == 0.] = float('inf')

        y_train  = (y_train - mean_y) / std_y
        y_val  = (y_val - mean_y) / std_y
        y_test  = (y_test - mean_y) / std_y

    dataset_train = TensorDataset(X_train, y_train)
    dataset_val = TensorDataset(X_val, y_val)
    dataset_test = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    return train_loader, val_loader, test_loader