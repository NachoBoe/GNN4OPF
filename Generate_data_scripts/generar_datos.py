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

# Levantar red

red = "30" # o "118"


if red == "30":
  net = pp.networks.case30()
else:
  net = pp.networks.case118()

# Permitir solo a los generadores cambiar su potencia al resolver opf
net.load['controllable'] = False
net.gen['controllable'] = True

# Empezar a generar datos
load_p_nom = net.load['p_mw']
load_q_nom = net.load['q_mvar']

X = []
Y = []
X_LOAD = []
# Dividir en 10 batches de 1000 datos
for j in range(10):
  for i in range(1000):
      y = np.zeros((len(net.bus["name"]),2))
      X_load = np.zeros((len(net.bus["name"]),2))
      print('iteracion:', i)
      # Tomar uniforme al rededor del valor nominal
      uniforme = np.random.uniform(0.7,1.3,size=len(load_p_nom))
      uniforme_react = np.random.uniform(0.7,1.3,size=len(load_q_nom))
      net.load['p_mw'] = uniforme*load_p_nom
      net.load['q_mvar'] = uniforme_react*load_q_nom

      # Resolver Flujo de carga
      pp.runpp(net,numba=False)
      X_i = np.array(net.res_bus.values)



      # Resolver OPF
      try:
        pp.runpm_ac_opf(net)
        y[net.gen["bus"]] = np.array([net.res_gen['p_mw'],net.res_gen['q_mvar']]).T
        y[net.ext_grid["bus"]] = np.array([net.res_ext_grid['p_mw'],net.res_ext_grid['q_mvar']]).T
        X_load[net.load["bus"]] = np.array([net.load['p_mw'],net.load['q_mvar']]).T
        Y.append(y)
        X.append(X_i)
        X_LOAD.append(X_load)
      except:
        pass
  X = np.array(X)
  Y = np.array(Y)
  X_LOAD = np.array(X_LOAD)
  np.save('./data_nueva/input'+red+f'_3_0/dato_{j}.npy',X)
  np.save('./data_nueva/p_opt'+red+f'_3_0/dato_{j}.npy',Y)
  np.save('./data_nueva/input_load'+red+f'_3_0/dato_{j}.npy',X_LOAD)



