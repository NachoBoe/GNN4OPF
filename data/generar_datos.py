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

import argparse

# Configurar argumentos de línea de comando
parser = argparse.ArgumentParser(description="Ejecutar simulación de pandapower con una red específica.")
parser.add_argument("red", type=str, choices=["30", "118"], help="Especificar el número de la red: '30' o '118'")
args = parser.parse_args()

# Usar el argumento red
red = args.red


if red == "30":
  net = pp.networks.case30()
  net.line["max_loading_percent"] *= 1.1

elif red == "118":
  net = pp.networks.case118()
  net.bus["max_vm_pu"] = 1.1
  net.bus["min_vm_pu"] = 0.9

# Permitir solo a los generadores cambiar su potencia al resolver opf
net.load['controllable'] = False
net.gen['controllable'] = True
pp.runpm_ac_opf(net)

gen_p_nom = deepcopy(net.res_gen.p_mw.values)
load_p_nom = deepcopy(net.load['p_mw'])
load_q_nom = deepcopy(net.load['q_mvar'])

net.bus["pm_param/setpoint_v"] = 1.0
# Empezar a generar datos

X = []
Y = []
X_LOAD = []
# Dividir en 10 batches de 1000 datos
for i in range(10000):
    y = np.zeros((len(net.bus["name"]),1))
    X_i = np.zeros((len(net.bus["name"]),3))
    print('iteracion:', i)
    # Tomar uniforme al rededor del valor nominal
    uniforme_activa_load = np.random.uniform(0.7,1.3,size=len(load_p_nom))
    uniforme_react_load = np.random.uniform(0.7,1.3,size=len(load_q_nom))
    uniforme_activa_gen = np.random.uniform(0.7,1.3,size=len(gen_p_nom))

    net.load.loc[:,'p_mw'] = uniforme_activa_load*load_p_nom
    net.load.loc[:,'q_mvar'] = uniforme_react_load*load_q_nom
    net.gen.loc[:,'p_mw'] =  uniforme_activa_gen*gen_p_nom
    net.gen.loc[:,"max_p_mw"] = net.gen['p_mw']
    net.gen.loc[:,"min_p_mw"] = net.gen['p_mw']

    # Resolver Flujo de carga
    pp.runpp(net,numba=False)
    X_i[net.gen["bus"]] += np.array([np.zeros(len(net.res_gen.p_mw)), np.zeros(len(net.res_gen.p_mw)),net.res_gen.p_mw]).T
    X_i[net.load["bus"]] += np.array([net.res_load.p_mw, net.res_load.q_mvar, np.zeros(len(net.res_load.q_mvar,))]).T

    # Resolver OPF
    try:
      pp.runpm_vstab(net)
      y[net.gen["bus"]] = net.res_gen.vm_pu.values.reshape(-1,1)
      Y.append(y)
      X.append(X_i)
    except:
      print("no convergio")
X = np.array(X)
Y = np.array(Y)
np.save(f'./red{red}/input_check.npy',X)
np.save(f'./red{red}/vm_pu_opt_check.npy',Y)



