
import psspy
import numpy as np
import pandas as pd
psspy.psseinit()

sav_file_path = './psse_scripts/DORAA_una_carga.sav'
ierr = psspy.case(sav_file_path)

load_P_data = pd.read_csv('LOADP_total_psse.csv')
load_P_data['Fecha'][1:] = pd.to_datetime(load_P_data['Fecha'][1:])
load_Q_data = pd.read_csv('LOADQ_total_psse.csv')
load_Q_data['Fecha'][1:] = pd.to_datetime(load_Q_data['Fecha'][1:])
gen_P_data = pd.read_csv('GENP_total_psse.csv')
gen_P_data['Fecha'][1:] = pd.to_datetime(gen_P_data['Fecha'][1:])
gen_Q_data = pd.read_csv('GENQ_total_psse.csv')
gen_Q_data['Fecha'][1:] = pd.to_datetime(gen_Q_data['Fecha'][1:])

# filter to only get data from 2022
load_P_data = load_P_data[1:][load_P_data['Fecha'][1:].dt.year == 2022].reset_index(drop=True)
load_Q_data = load_Q_data[1:][load_Q_data['Fecha'][1:].dt.year == 2022].reset_index(drop=True)
gen_P_data = gen_P_data[1:][gen_P_data['Fecha'][1:].dt.year == 2022].reset_index(drop=True)
gen_Q_data = gen_Q_data[1:][gen_Q_data['Fecha'][1:].dt.year == 2022].reset_index(drop=True)

# # Get list of columns
columns_load_P = load_P_data.columns.tolist()
columns_load_Q = load_Q_data.columns.tolist()
comumns_gen_P = gen_P_data.columns.tolist()
columns_gen_Q = gen_Q_data.columns.tolist()
columns_load_P.remove('Fecha')
columns_load_Q.remove('Fecha')
comumns_gen_P.remove('Fecha')
columns_gen_Q.remove('Fecha')

# Values of default loads for buses that dont have Qload
ierr, [load_numbers] = psspy.aloadint(sid=-1, flag=4, string="NUMBER")
ierr, [loads] = psspy.aloadcplx(sid=-1, flag=4, string="TOTALACT") # si P es negativa va a 0 todo
Qloads_dict = {}
for load_number, load_value in zip(load_numbers, loads):
    Qloads_dict[str(load_number)] = load_value.imag

# Values of default gens for buses that dont have Qgen
ierr, [machine_numbers] = psspy.amachint(sid=-1, flag=4, string="NUMBER")
ierr, [machineIDs] = psspy.amachchar(sid=-1, flag=4, string="ID")
ierr, [PQgen] = psspy.amachcplx(sid=-1, flag=4, string="PQGEN")
Qgen_dict = {}
for load_number, machine_id, load_value in zip(machine_numbers, machineIDs, PQgen):
    strg = str(load_number)+'_'+machine_id.replace(" ", "")
    Qgen_dict[strg] = load_value.imag

for index in range(load_P_data.shape[0])[10000:]:
    for column in columns_load_P:
        Pload = load_P_data.loc[index,column]
        if column in columns_load_Q:
            Qload = load_Q_data.loc[index,column]
        else:
            Qload = Qloads_dict[column]
            Qload *= np.random.uniform(0.8, 1.2)
        ierr = psspy.load_data_3(int(column), "1", realar1=Pload, realar2=Qload)
    for column in comumns_gen_P:
        bus_number,id_gen = column.split('_')
        Pgen = gen_P_data.loc[index,column]
        if column in columns_gen_Q:
            Qgen = gen_Q_data.loc[index,column]
        else:
            Qgen = Qgen_dict[column]
            Qgen *= np.random.uniform(0.8, 1.2) # si la Q gen no esta en datos UTE poner uniforme del valor default en psse
        ierr = psspy.machine_data_2(int(bus_number), id_gen, realar1=Pgen, realar2=Qgen)

    # Run power flow
    # ierr = psspy.fnsl()
    # ierr, [voltajes,angles] = psspy.abusreal(-1,2,string=["PU","ANGLED"]) 

    # Run OPF
    ierr = psspy.nopf(0,1)

    # Save data in npy file
    break

