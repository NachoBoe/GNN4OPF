import torch
import numpy as np
import torch.nn as nn
import pandapower as pp
import tqdm

import julia
julia.install()

from julia.api import Julia
jl = Julia(compiled_modules=False)


class NormalizedError(nn.Module):
    def __init__(self):
        super(NormalizedError, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calcula el error normalizado entre la predicción y el valor real.

        Parámetros:
        y_pred (Tensor): Predicciones del modelo.
        y_true (Tensor): Valores reales.

        Retorna:
        Tensor: Error normalizado.
        """
        # Asegurarse de que las predicciones y los valores reales están en la CPU
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()

        # Calcular la norma de la diferencia entre la predicción y el valor real
        numerator = torch.norm(y_pred - y_true,dim=1)

        # Calcular la norma del valor real
        denominator = torch.norm(y_true,dim=1)

        # # Evitar la división por cero
        # if denominator == 0:
        #     return torch.tensor(0.0)

        # Calcular y retornar el error normalizado
        return torch.mean(torch.sqrt(numerator / denominator))


class PlossMetric(nn.Module):
    def __init__(self, net):
        super(PlossMetric, self).__init__()
        self.net = net


    def forward(self, X, y):
        X_np = X.clone().detach().cpu().numpy()
        y_np = y.clone().detach().cpu().numpy()
        loss = 0
        converged = 0
        for i in range(X.shape[0]):
            # se agrega para que ande en la red de uru, tb anda en ieee. Lo de arriba era lo de antes
            id_load = [j for j, num in enumerate(self.net.bus.reset_index()['index'].to_list()) if num in self.net.load.bus.to_list()]
            id_gen = [j for j, num in enumerate(self.net.bus.reset_index()['index'].to_list()) if num in self.net.gen.bus.to_list()]
            # print("id_gen",id_gen)
            # print("id_load",id_load)
            # print("X_np",X_np[i])
            # print("y_np",y_np[i,id_gen][:,0])
            # print("net.gen",self.net.gen)
            # print('load p',X_np[i,id_load,0].shape)
            # print('load q',X_np[i,id_load,1].shape)
            self.net.load.loc[:,'p_mw'] = X_np[i,id_load,0]
            self.net.load.loc[:,'q_mvar'] = X_np[i,id_load,1]
            self.net.gen.loc[:,'p_mw'] =  X_np[i,id_gen,2]
            # print('gen p',X_np[i,id_gen,2].shape)
            self.net.gen.loc[:,'vm_pu'] =  y_np[i,id_gen][:,0]
            
            # print('vm_pu',y_np[i,id_gen,0].shape)
            # print("net.gen After",self.net.gen)

            try:
                pp.runpp(self.net, numba=False)
                loss += self.net.res_line.pl_mw.sum()
                converged += 1
            except:
                pass
        if converged!=0:
            loss /= converged
        else:
            loss = -1
        return loss

def relative_feas_error(x,x_min,x_max):
    error = np.max((np.zeros_like(x),x-x_max),axis=0) - np.min((np.zeros_like(x),x-x_min),axis=0)
    return (error/(x_max-x_min)).sum()

def feas_and_volt_metric(model,val_loader,net):

    idxs_gen = net.bus.index.get_indexer(list(net.gen.bus.values))
    idxs_load = net.bus.index.get_indexer(list(net.load.bus.values))

    feasibilty_metric = 0
    v_setpoint_metric = 0
    no_conv_count = 0

    for x in val_loader:
        output = model(x[0]).detach().cpu()
        p_ext_grid, q_gen, vm_pu_gen, ang_gen = output[:,:,0], output[:,:,1], output[:,:,2], output[:,:,3]
        batch_size = vm_pu_gen.shape[0]
        feas_count = 0
        for i in range(batch_size):
            net.gen.vm_pu = vm_pu_gen[i][idxs_gen].detach().cpu().numpy()
            net.load.p_mw = x[0][i,idxs_load,0].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            net.load.q_mvar = x[0][i,idxs_load,1].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            net.gen.p_mw = x[0][i,idxs_gen,2].detach().cpu().numpy() * 100 # para pasar los voltajes a valores no pu
            try:
                pp.runpp(net,numba=False)

                lineas_cargadas = relative_feas_error(net.res_line.loading_percent,0,100)
                trafos_cargados = relative_feas_error(net.res_trafo.loading_percent,0,100)
                gen_q = relative_feas_error(net.res_gen.q_mvar,net.gen.min_q_mvar,net.gen.max_q_mvar)
                ext_grid_q = relative_feas_error(net.res_ext_grid.q_mvar,net.ext_grid.min_q_mvar,net.ext_grid.max_q_mvar)
                ext_grid_p = relative_feas_error(net.res_ext_grid.p_mw,net.ext_grid.min_p_mw,net.ext_grid.max_p_mw)
                vmpu = relative_feas_error(net.res_bus.vm_pu,net.bus.min_vm_pu,net.bus.max_vm_pu)
                feasibilty_metric += (lineas_cargadas + trafos_cargados + gen_q + vmpu + ext_grid_p + ext_grid_q)

                v_setpoint_metric += np.abs(net.res_bus.vm_pu.values - 1).sum()
                feas_count += 1
            except:
                no_conv_count += 1
        if feas_count == 0:
            feasibilty_metric += 1000000
            v_setpoint_metric += 1000000
        else:
            feasibilty_metric /= feas_count
            v_setpoint_metric /= feas_count

    return feasibilty_metric/len(val_loader), v_setpoint_metric/len(val_loader), no_conv_count

