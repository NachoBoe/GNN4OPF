
import torch
import numpy as np
# import pandapower as pp

def get_max_q(net):
  q_min = torch.zeros(len(net.bus)) 
  q_max = torch.zeros(len(net.bus))
  for idx in range(len(net.gen)):
    q_min[net.gen.iloc[idx].bus] = net.gen.iloc[idx].min_q_mvar
    q_max[net.gen.iloc[idx].bus] = net.gen.iloc[idx].max_q_mvar
  for idx in range(len(net.ext_grid)):
    q_min[net.ext_grid.iloc[idx].bus] = net.ext_grid.iloc[idx].min_q_mvar
    q_max[net.ext_grid.iloc[idx].bus] = net.ext_grid.iloc[idx].max_q_mvar
  return q_min/ net.sn_mva,q_max/net.sn_mva

def get_max_min_values(net,device):
    N = len(net.bus.index)

    min_voltages_pu = torch.ones(N)*0.8
    max_voltages_pu = torch.ones(N)*1.2

    min_angles = -torch.ones(N) * np.pi / 2
    max_angles = torch.ones(N)* np.pi / 2

    min_p = -torch.zeros(N) 
    max_p = torch.zeros(N)

    min_q_gen,max_q_gen = get_max_q(net)

    # Fix for ext grid
    min_voltages_pu[net.ext_grid["bus"].values.astype(int)] = 1.0
    max_voltages_pu[net.ext_grid["bus"].values.astype(int)] = 1.0
    min_angles[net.ext_grid["bus"].values.astype(int)] = 0.0
    max_angles[net.ext_grid["bus"].values.astype(int)] = 0.0
    min_p[net.ext_grid["bus"].values.astype(int)] = torch.Tensor(net.ext_grid["min_p_mw"].values)
    max_p[net.ext_grid["bus"].values.astype(int)] = torch.Tensor(net.ext_grid["max_p_mw"].values)

    min_vector = torch.vstack([min_p, min_q_gen, min_voltages_pu, min_angles]).t().to(device)
    max_vector = torch.vstack([max_p, max_q_gen, max_voltages_pu, max_angles]).t().to(device)

    return min_vector, max_vector


def constraint_violation_power_flow(V_mag,V_ang,p,q,Y_bus):
  V = V_mag * (torch.cos(V_ang) + torch.sin(V_ang)*1j)
  S = p + 1j * q
  AC_equality = S - torch.bmm(torch.diag_embed(V), torch.conj(torch.bmm(Y_bus.repeat(V.shape[0],1,1),V.unsqueeze(2)))).squeeze()
  return AC_equality

def equality_penalty(U):
    Y = U**2
    return Y

def cost_function_voltage(V_mag):
  ''' U is the output of the GNN, BxNx3 (Qgen, V, angle)
      a and b Nx3 are the upper and lower limits for the three magnitudes '''
  cost = (V_mag - 1)**2
  return torch.sum(cost,axis=-1)

def my_loss(U,X,Y_line,Y_bus,ika_max,dual_variables):
  dual_acflow_real = dual_variables[0]
  dual_acflow_imag = dual_variables[1]
  dual_lines = dual_variables[2]

  U = U[0]
  p_load = U[:,:,0]
  q_load = U[:,:,1]
  p_gen = U[:,:,2]
  p_ext_grid = X[:,:,0]
  q_gen = X[:,:,1]
  V_mag = X[:,:,2]
  delta = X[:,:,3]
  p = p_gen + p_ext_grid - p_load  
  q = q_gen - q_load

  
  # AC flow penalty
  AC_flow = constraint_violation_power_flow(V_mag, delta, p, q, Y_bus)

  AC_flow_penalty_real = equality_penalty(torch.real(AC_flow))
  AC_flow_penalty_imag = equality_penalty(torch.imag(AC_flow))

  # Objective cost
  objective = cost_function_voltage(V_mag)

  # Sum of all penalties
  loss =  objective  + torch.mv(AC_flow_penalty_real,dual_acflow_real) +  torch.mv(AC_flow_penalty_imag,dual_acflow_imag) # + torch.mv(Sij_penalty,dual_lines)  

  loss = torch.mean(loss)
  return loss


