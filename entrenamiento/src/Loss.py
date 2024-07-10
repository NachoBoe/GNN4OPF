
import torch
import numpy as np
# import pandapower as pp


# That also works for a list of buses
def bus_pos(buses, net):
  N = len(net.bus.index)
  bus_map = {net.bus.index[i]: i for i in range(N)}
  try:
      return [bus_map[bus] for bus in buses]
  except:
      return bus_map[buses]

def get_max_q(net):
  q_min = torch.zeros(len(net.bus)) 
  q_max = torch.zeros(len(net.bus))
  for idx in range(len(net.gen)):
    q_min[bus_pos(net.gen.iloc[idx].bus, net)] = net.gen.iloc[idx].min_q_mvar
    q_max[bus_pos(net.gen.iloc[idx].bus, net)] = net.gen.iloc[idx].max_q_mvar
  for idx in range(len(net.ext_grid)):
    q_min[bus_pos(net.ext_grid.iloc[idx].bus, net)] = net.ext_grid.iloc[idx].min_q_mvar
    q_max[bus_pos(net.ext_grid.iloc[idx].bus, net)] = net.ext_grid.iloc[idx].max_q_mvar
  return q_min/ net.sn_mva,q_max/net.sn_mva

def get_max_min_values(net,device):
    N = len(net.bus.index)

    min_voltages_pu = torch.ones(N)*0.8
    max_voltages_pu = torch.ones(N)*1.2

    min_shunt_q = torch.zeros(N)
    max_shunt_q = torch.zeros(N)

    min_angles = -torch.ones(N) * np.pi / 2
    max_angles = torch.ones(N)* np.pi / 2

    min_p = -torch.zeros(N) 
    max_p = torch.zeros(N)

    min_q_gen,max_q_gen = get_max_q(net)
    
    min_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].min_q_mvar.values).float() /net.sn_mva
    max_shunt_q[bus_pos(net.sgen.loc[net.sgen.controllable==True]["bus"].values.astype(int), net)] = torch.from_numpy(net.sgen.loc[net.sgen.controllable==True].max_q_mvar.values).float() /net.sn_mva


    # Fix for ext grid
    min_voltages_pu[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 1.0
    max_voltages_pu[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 1.0
    min_angles[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 0.0
    max_angles[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 0.0
    min_p[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = -10
    max_p[bus_pos(net.ext_grid["bus"].values.astype(int), net)] = 10

    min_vector = torch.vstack([min_p, min_q_gen, min_shunt_q, min_voltages_pu, min_angles]).t().to(device)
    max_vector = torch.vstack([max_p, max_q_gen, max_shunt_q, max_voltages_pu, max_angles]).t().to(device)

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
  p_sgen = U[:,:,3]

  p_ext_grid = X[:,:,0]
  q_gen = X[:,:,1]
  q_shunt = X[:,:,2]
  V_mag = X[:,:,3]
  delta = X[:,:,4]
  p = p_gen + p_ext_grid + p_sgen - p_load  
  q = q_gen + q_shunt - q_load 
  
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


