
import psspy
import numpy as np
import pandas
psspy.psseinit()

sav_file_path = './RedSimplificada_DORAA.sav'
ierr = psspy.case(sav_file_path)

ierr, [bus_numbers] = psspy.abusint(string=["NUMBER"])

ierr, [load_numbers] = psspy.aloadint(sid=-1, flag=4, string="NUMBER")
ierr, [loadIDs] = psspy.aloadchar(sid=-1, flag=4, string="ID")
ierr, [loads] = psspy.aloadcplx(sid=-1, flag=4, string="TOTALACT") # si P es negativa va a 0 todo
# ierr, [loads] = psspy.aloadreal(sid=-1, flag=4, string="TOTALACT") # Solo la activa, si P es negativa va a 0

ierr, [machine_numbers] = psspy.amachint(sid=-1, flag=4, string="NUMBER")
ierr, [machineIDs] = psspy.amachchar(sid=-1, flag=4, string="ID")
ierr, [Pgen] = psspy.amachreal(sid=-1, flag=4, string="PGEN")


# Set load values for each bus
for i, load_number in enumerate(load_numbers):
    load_id = loadIDs[i]
    load = loads[i]
    ierr = psspy.load_data_3(load_number, load_id, realar1=load.real, realar2=load.imag)

for i, machine_number in enumerate(machine_numbers):
    machine_id = machineIDs[i]
    pgen = Pgen[i]
    ierr = psspy.machine_data_2(machine_number, machine_id, realar1=pgen)

# Run power flow
ierr = psspy.fnsl()
ierr, [voltajes,angles] = psspy.abusreal(-1,2,string=["PU","ANGLED"])

# Save the modified case
# ierr = psspy.save("modified_case.sav")
