
import psspy
import numpy as np
import pandas
psspy.psseinit()

sav_file_path = './DORAA_una_carga.sav'
ierr = psspy.case(sav_file_path)

load_P_data = pandas.read_csv('./scripts/datos_ute_para_psse/PLOAD.csv')
load_Q_data = pandas.read_csv('./scripts/datos_ute_para_psse/QLOAD.csv')


# Get list of columns
columns_load = load_P_data.columns.tolist()
for index, row in load_P_data.iterrows(): # iterar sobre el indice de cada fila y ahi ir y buscar en cada csv, cambiar como esta ahora
    for column in columns_load:
        Pload = row[column]
        Qload = _
        ierr = psspy.load_data_3(int(column), "1", realar1=Pload, realar2=Qload)
    break

ierr, [machine_numbers] = psspy.amachint(sid=-1, flag=4, string="NUMBER")
ierr, [machineIDs] = psspy.amachchar(sid=-1, flag=4, string="ID")
ierr, [Pgen] = psspy.amachreal(sid=-1, flag=4, string="PGEN")

for i, machine_number in enumerate(machine_numbers):
    machine_id = machineIDs[i]
    pgen = Pgen[i]
    qgen = 0 # poner la qgen de los datos de la ute
    ierr = psspy.machine_data_2(machine_number, machine_id, realar1=pgen, realar2=qgen)

# Run power flow
ierr = psspy.fnsl()
ierr, [voltajes,angles] = psspy.abusreal(-1,2,string=["PU","ANGLED"])

# Save the modified case
# ierr = psspy.save("modified_case.sav")
