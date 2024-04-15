
import psspy
import numpy as np
import pandas
psspy.psseinit()

sav_file_path = './DORAA_una_carga.sav'
ierr = psspy.case(sav_file_path)

load_P_data = pandas.read_csv('./scripts/datos_ute_para_psse/PLOAD.csv')
load_Q_data = pandas.read_csv('./scripts/datos_ute_para_psse/QLOAD.csv')
gen_P_data = pandas.read_csv('./scripts/datos_ute_para_psse/PGEN.csv')
gen_Q_data = pandas.read_csv('./scripts/datos_ute_para_psse/QGEN.csv')

# Get list of columns
columns_load_P = load_P_data.columns.tolist()
columns_load_Q = load_Q_data.columns.tolist()
comumns_gen_P = gen_P_data.columns.tolist()
columns_gen_Q = gen_Q_data.columns.tolist()
cant_datos = len(columns_load_P)

# Values of default loads for buses that dont have Qload
ierr, [load_numbers] = psspy.aloadint(sid=-1, flag=4, string="NUMBER")
ierr, [loads] = psspy.aloadcplx(sid=-1, flag=4, string="TOTALACT") # si P es negativa va a 0 todo
Qloads_dict = {}
for load_number, load_value in zip(load_numbers, loads):
    Qloads_dict[load_number] = load_value.imag

# Values of default gens for buses that dont have Qgen
# ierr, [machine_numbers] = psspy.amachint(sid=-1, flag=4, string="NUMBER")
# ierr, [machineIDs] = psspy.amachchar(sid=-1, flag=4, string="ID")
# ierr, [Pgen] = psspy.amachreal(sid=-1, flag=4, string="PGEN")

for index in range(cant_datos):
    for column in columns_load_P:
        Pload = load_P_data[index][column]
        if column in columns_load_Q:
            Qload = load_Q_data[index][column]
        else:
            Qload = Qloads_dict[column]
            Qload *= np.random.uniform(0.8, 1.2)
        ierr = psspy.load_data_3(int(column), "1", realar1=Pload, realar2=Qload)
    for column in comumns_gen_P:
        bus_number,id_gen = column.split('_')
        Pgen = gen_P_data[index][bus_number]
        if column in columns_gen_Q:
            Qgen = gen_Q_data[index][bus_number]
        else:
            Qgen = 0 # si la Q gen no esta en datos UTE poner uniforme del valor default en psse
        ierr = psspy.machine_data_2(int(bus_number), id_gen, realar1=Pgen, realar2=Qgen)

    # Run power flow
    ierr = psspy.fnsl()
    ierr, [voltajes,angles] = psspy.abusreal(-1,2,string=["PU","ANGLED"]) 

    # Run OPF

    # Save data in  npy file
    break

