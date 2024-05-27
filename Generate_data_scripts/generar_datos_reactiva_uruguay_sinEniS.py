import pandapower as pp
import pandapower.networks as nw
from copy import deepcopy
import pandapower.plotting as plot
import numpy as np
import pandas as pd
import os
import numba

import julia
julia.install()

from julia.api import Julia
jl = Julia(compiled_modules=False)

import warnings
warnings.filterwarnings('ignore')


gen_p = pd.read_csv("GENP_total_pandapower.csv")
load_p = pd.read_csv("LOADP_total_pandapower.csv")
net = pp.from_pickle('uy_pp_net_v12.p')

net.bus["pm_param/setpoint_v"] = 1.0
net.sn_mva = 100
# Empezar a generar datos

X = []
Y_volt = []
Y_react = []
Y_switch_shunts = []
# Dividir en 10 batches de 1000 datos
buses_solar = [92040,92030,92020,92070,92640,92020, 92060, 92000]
buses_eolico = [92710,92091,92760,92440,92611,92700,92590,92061,92610,92580,92790,92485,92420,92620,92910,92524,92411,92490] #90160

for index in range(21000,gen_p.shape[0]):
  net = pp.from_pickle('uy_pp_net_v12.p')
  net.bus["pm_param/setpoint_v"] = 1.0
  net.sn_mva = 100
  net.line.max_loading_percent = 110

  y_volt = np.zeros((len(net.bus["name"]),1))
  y_react = np.zeros((len(net.bus["name"]),1))
  y_switch_shunts  = np.zeros((len(net.bus["name"]),1))
  X_i = np.zeros((len(net.bus["name"]),4))
  print('iteracion:', index)

  net.gen.p_mw = gen_p.iloc[index,1:].sort_index().values
  net.load.p_mw = load_p.iloc[index,1:].sort_index().values
  net.load.q_mvar = load_p.iloc[index,1:].sort_index().values * np.tan(np.arccos(0.92))
  for idx, g in net.gen.iterrows():
    if g.bus in buses_solar:
        pp.create_sgen(net, g.bus,g.p_mw,controllable=False)
        net.gen.drop(idx,inplace=True)
    if g.bus in buses_eolico:
        pp.create_sgen(net, g.bus,g.p_mw,controllable=False)
        net.gen.drop(idx,inplace=True)

  # fijamos la activa para que no cambie
  net.gen.loc[:,"max_p_mw"] = net.gen['p_mw']
  net.gen.loc[:,"min_p_mw"] = net.gen['p_mw']

  # Rellenar X
  X_i[net.bus.index.get_indexer(list(net.gen.bus.values))] += np.array([np.zeros(len(net.gen.p_mw)), np.zeros(len(net.gen.p_mw)),net.gen.p_mw.values.astype('float64'), np.zeros(len(net.gen.p_mw))]).T
  X_i[net.bus.index.get_indexer(list(net.load.bus.values))] += np.array([net.load.p_mw.values.astype('float64'), net.load.q_mvar.values.astype('float64'), np.zeros(len(net.load.q_mvar)), np.zeros(len(net.load.q_mvar))]).T
  X_i[net.bus.index.get_indexer(list(net.sgen.bus.values))] += np.array([np.zeros(len(net.sgen.p_mw)),np.zeros(len(net.sgen.p_mw)),np.zeros(len(net.sgen.p_mw)), net.sgen.p_mw.values.astype('float64')]).T

  # Resolver OPF
  try:
    pp.runpm_vstab(net)
    #Rellenar y
    y_volt[net.bus.index.get_indexer(list(net.gen.bus.values))] = net.res_gen.vm_pu.values.reshape(-1,1)
    y_react[net.bus.index.get_indexer(list(net.gen.bus.values))] = net.res_gen.q_mvar.values.reshape(-1,1)
    y_switch_shunts[net.bus.index.get_indexer(list(net.sgen.bus.values))] = net.res_sgen.q_mvar.values.reshape(-1,1)
    y_switch_shunts[net.bus.index.get_indexer(net.sgen.bus.iloc[np.where(net.sgen.controllable==False)[0]])] = 0
    Y_volt.append(y_volt)
    Y_react.append(y_react)
    Y_switch_shunts.append(y_switch_shunts)
    X.append(X_i)
  except:
    print("no convergio")

  if (index+1)%1000 == 0:
    X = np.array(X)
    Y_volt = np.array(Y_volt)
    Y_react = np.array(Y_react)
    Y_switch_shunts = np.array(Y_switch_shunts)
    np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/input_{np.ceil((index+1)/1000)}.npy',X)
    np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/vm_pu_opt_{np.ceil((index+1)/1000)}.npy',Y_volt)
    np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/q_mvar_opt_{np.ceil((index+1)/1000)}.npy',Y_react)
    np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/q_switch_shunt_opt_{np.ceil((index+1)/1000)}.npy',Y_switch_shunts)
    print('Guardado')
    X = []
    Y_volt = []
    Y_react = []
    Y_switch_shunts = []


X = np.array(X)
Y_volt = np.array(Y_volt)
Y_react = np.array(Y_react)
Y_switch_shunts = np.array(Y_switch_shunts)
np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/input_{np.ceil((index+1)/1000)}.npy',X)
np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/vm_pu_opt_{np.ceil((index+1)/1000)}.npy',Y_volt)
np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/q_mvar_opt_{np.ceil((index+1)/1000)}.npy',Y_react)
np.save(f'./data_reactiva_uruguay_sin_eolicanisolar_v4/q_switch_shunt_opt_{np.ceil((index+1)/1000)}.npy',Y_switch_shunts)

