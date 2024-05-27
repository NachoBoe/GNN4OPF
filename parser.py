import numpy as np
import pandas as pd


### BUS DATA

def bus_data_df(archivo, limpio = True):
    '''
    Devuelve un dataframe con bus data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF SYSTEM-WIDE DATA, BEGIN BUS DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF BUS DATA, BEGIN LOAD DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_bus = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'NAME', 'BASKV', 'IDE', 'AREA', 'ZONE', 'OWNER', 'VM', 'VA', 'NVHI', 'NVLO', 'EVHI', 'EVLO']

    # Crea el DataFrame
    bus_data = pd.DataFrame(datos_bus, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['BASKV', 'I', 'AREA', 'ZONE', 'VM', 'VA', 'NVHI', 'NVLO', 'EVHI', 'EVLO', 'IDE']
    bus_data[columnas_numericas] = bus_data[columnas_numericas].apply(pd.to_numeric)

    # Dejando limpio como lo queremos
    if limpio:
        # bus_data = bus_data[['NAME', 'BASKV', 'VM', 'VA', 'IDE']]
        bus_data = bus_data.set_index('NAME')
        bus_data.index = bus_data.index.str.replace('\'', '', regex=False)
        bus_data.index = bus_data.index.str.replace(' ', '', regex=False)
        bus_data['V (KV)'] = bus_data['BASKV'] * bus_data['VM'] * (np.cos(np.deg2rad(bus_data['VA'])) + np.sin(np.deg2rad(bus_data['VA'])) * 1j)

    return bus_data



### LOAD DATA

def load_data_df(archivo, limpio=True):
    '''
    Devuelve un dataframe con load data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF BUS DATA, BEGIN LOAD DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_carga = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'ID','STAT','AREA','ZONE', 'PL', 'QL', 'IP', 'IQ', 'YP','YQ', 'OWNER','SCALE','INTRPT',  'DGENP', 'DGENQ', 'DGENF']
    # Crea el DataFrame
    load_data = pd.DataFrame(datos_carga, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['PL', 'QL', 'IP', 'IQ', 'YP','YQ','SCALE','INTRPT',  'DGENP', 'DGENQ', 'DGENF','STAT']
    load_data[columnas_numericas] = load_data[columnas_numericas].apply(pd.to_numeric)

    # Dejando limpio como lo queremos
    if limpio:
        # load_data = load_data[['I', 'PL', 'QL']]
        # load_data = load_data.groupby('I')[['PL', 'QL']].sum().reset_index()
        load_data = load_data.set_index('I')
        load_data.index = load_data.index.str.replace('150.00', '', regex=False)
        load_data.index = load_data.index.str.replace('500.00', '', regex=False)
        load_data.index = load_data.index.str.replace(' ', '', regex=False)
        load_data.index = load_data.index.str.replace('\'', '', regex=False)
        load_data['SL'] = load_data['PL'] + load_data['QL'] * 1j
    return load_data


### FIXED SHUNT DATA

def fixed_shunt_data_df(archivo):
    '''
    Devuelve un dataframe con shunt data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_shunt = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I','ID','STATUS',  'GL', 'BL']
    # Crea el DataFrame
    shunt_data = pd.DataFrame(datos_shunt, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['GL', 'BL']
    shunt_data[columnas_numericas] = shunt_data[columnas_numericas].apply(pd.to_numeric)

    # Muestra el DataFrame resultante
    return shunt_data


### GENERATOR DATA

def generator_data_df(archivo, limpio = True):
    '''
    Devuelve un dataframe con shunt data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_gen = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'ID', 'PG', 'QG', 'QT', 'QB', 'VS', 'IREG', 'MBASE', 'ZR', 'ZX', 'RT', 'XT', 'GTAP', 'STAT', 'RMPCT', 'PT', 'PB', 'O1', 'F1']
    # Crea el DataFrame
    gen_data = pd.DataFrame(datos_gen, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['PG', 'QG', 'QT', 'QB', 'VS', 'MBASE', 'ZR', 'ZX', 'RT', 'XT', 'GTAP', 'RMPCT', 'PT', 'PB', 'O1']
    gen_data[columnas_numericas] = gen_data[columnas_numericas].apply(pd.to_numeric)

    # Muestra el DataFrame resultante
    if limpio:
        # gen_data = gen_data[['I', 'PG', 'QG']]
        # gen_data = gen_data.groupby('I')[['PG', 'QG']].sum().reset_index()
        gen_data = gen_data.set_index('I')
        gen_data.index = gen_data.index.str.replace('150.00', '', regex=False)
        gen_data.index = gen_data.index.str.replace(' ', '', regex=False)
        gen_data.index = gen_data.index.str.replace('500.00', '', regex=False)
        gen_data.index = gen_data.index.str.replace('\'', '', regex=False)
        gen_data['SG'] = gen_data['PG'] + gen_data['QG'] * 1j
    return gen_data


### BRANCH DATA

def branch_data_df(archivo, limpio = True):
    '''
    Devuelve un dataframe con branch data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF GENERATOR DATA, BEGIN BRANCH DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF BRANCH DATA, BEGIN SYSTEM SWITCHING DEVICE DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_branch = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'J', 'CKT', 'R', 'X', 'B', 'NAME', 'RATE1', 'RATE2', 'RATE3', 'RATE4', 'RATE5', 'RATE6', 'RATE7', 'RATE8', 'RATE9', 'RATE10', 'RATE11', 'RATE12', 'GI', 'BI', 'GJ', 'BJ', 'STAT', 'MET', 'LEN', 'O1', 'F1']
    # Crea el DataFrame
    branch_data = pd.DataFrame(datos_branch, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['R', 'X', 'B', 'BI', 'BJ', 'GI', 'GJ']
    branch_data[columnas_numericas] = branch_data[columnas_numericas].apply(pd.to_numeric)


    if limpio:
        # branch_data = branch_data[['I', 'J', 'R', 'X', 'B']]
        branch_data['BASKV'] = branch_data['I'].str.split().str[-1].str.extract(r'(\d+)').astype(int)
        branch_data['I'] = branch_data['I'].str.replace('150.00', '', regex=False)
        branch_data['I'] = branch_data['I'].str.replace(' ', '', regex=False)
        branch_data['I'] = branch_data['I'].str.replace('500.00', '', regex=False)
        branch_data['I'] = branch_data['I'].str.replace('\'', '', regex=False)

        branch_data['J'] = branch_data['J'].str.replace('150.00', '', regex=False)
        branch_data['J'] = branch_data['J'].str.replace(' ', '', regex=False)
        branch_data['J'] = branch_data['J'].str.replace('500.00', '', regex=False)
        branch_data['J'] = branch_data['J'].str.replace('\'', '', regex=False)

    return branch_data


### TRAFO DATA

def trafo_data_df(archivo):
    '''
    Devuelve un dataframe con trafo data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF SYSTEM SWITCHING DEVICE DATA, BEGIN TRANSFORMER DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF TRANSFORMER DATA, BEGIN AREA DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_trafos = [','.join([linea.replace('\n', '') for linea in lineas[i:i+4]]).split(',') for i in range(0, len(lineas), 4)]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'J', 'K', 'CKT', 'CW', 'CZ', 'CM', 'MAG1', 'MAG2', 'NMETR', 'NAME', 'STAT', 'F1', 'O1', 'O2', 'F2', 'O3', 'F3', 'O4', 'F4', 'VECGRP', 'R1-2', 'X1-2', 'SBASE1-2', 'WINDV1', 'NOMV1', 'ANG1', 'RATE1-1', 'RATE1-2', 'RATE1-3', 'RATE1-4', 'RATE1-5', 'RATE1-6', 'RATE1-7', 'RATE1-8', 'RATE1-9', 'RATE1-10', 'RATE1-11', 'RATE1-12', 'COD1', 'CONT1', 'RMA1', 'RMI1', 'VMA1', 'VMI1', 'NTP1', 'TAB1', 'CR1', 'CX1', 'CNXA1', 'NOD1', 'WINDV2', 'NOMV2']

    # Crea el DataFrame
    trafo_data = pd.DataFrame(datos_trafos, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    trafo_data['CKT'] = trafo_data['CKT'].str.replace(' ', '', regex=False)
    trafo_data['CKT'] = trafo_data['CKT'].str.replace('\'', '', regex=False)
    columnas_numericas = ['K', 'CKT', 'CW', 'CM', 'MAG1', 'MAG2', 'NMETR','STAT', 'F1', 'O1', 'O2', 'F2', 'O3', 'F3', 'O4', 'F4', 'R1-2', 'X1-2', 'SBASE1-2', 'WINDV1', 'NOMV1', 'ANG1', 'RATE1-1', 'RATE1-2', 'RATE1-3', 'RATE1-4', 'RATE1-5', 'RATE1-6', 'RATE1-7', 'RATE1-8', 'RATE1-9', 'RATE1-10', 'RATE1-11', 'RATE1-12', 'COD1', 'RMA1', 'RMI1', 'VMA1', 'VMI1', 'NTP1', 'TAB1', 'CR1', 'CX1', 'CNXA1', 'NOD1', 'WINDV2', 'NOMV2']
    trafo_data[columnas_numericas] = trafo_data[columnas_numericas].apply(pd.to_numeric)

    return trafo_data


### SWITCHED SHUNT DATA

def switched_shunt_data_df(archivo):
    '''
    Devuelve un dataframe con shunt data a partir del archivo
    '''
    # Abre el archivo en modo binario y decodifica con la codificación adecuada
    with open(archivo, 'rb') as archivo:
        contenido = archivo.read().decode('ISO-8859-1')  # Cambia ISO-8859-1 a la codificación correcta si es diferente

    # Divide el contenido en líneas
    lineas = contenido.split('\n')

    # Encuentra la posición de "END OF SYSTEM-WIDE DATA"
    indice_inicio = np.where(np.array(lineas) == '0 / END OF FACTS DEVICE DATA, BEGIN SWITCHED SHUNT DATA\r')[0][0]
    indice_fin = np.where(np.array(lineas) == '0 / END OF SWITCHED SHUNT DATA, BEGIN GNE DATA\r')[0][0]

    # Elimina las líneas vacías y las líneas de comentario
    lineas = [linea.strip() for linea in lineas[indice_inicio + 1:indice_fin] if linea.strip() and not linea.startswith('@!')]

    # Divide las líneas en columnas usando espacios en blanco como separador
    datos_shunt = [linea.split(',') for linea in lineas]

    # Define los nombres de las columnas
    nombres_columnas = ['I', 'MODSW', 'ADJM', 'ST', 'VSWHI', 'VSWLO', 'SWREG', 'RMPCT', 'RMIDNT', 'BINIT', 'N1', 'B1']
    # Crea el DataFrame
    shunt_data = pd.DataFrame(datos_shunt, columns=nombres_columnas)

    # Convierte las columnas numéricas a tipos numéricos
    columnas_numericas = ['MODSW', 'ADJM', 'ST', 'VSWHI', 'VSWLO', 'RMPCT', 'N1', 'B1', 'BINIT']
    shunt_data[columnas_numericas] = shunt_data[columnas_numericas].apply(pd.to_numeric).replace(np.nan, 0)



    # Muestra el DataFrame resultante
    return shunt_data



def data(archivo):
    '''
    Devuelve todos los nodos de bus data con potencia generada/consumida
    '''
    bus_data = bus_data_df(archivo)
    load_data = load_data_df(archivo)
    gen_data = generator_data_df(archivo)
    nodos = pd.DataFrame(bus_data)
    nodos['SL'] = 0
    nodos['SG'] = 0
    
    # Asigna los valores de 'SL' de load_data a 'SL' de nodos
    nodos.loc[load_data.index, 'SL'] = load_data['SL']
    nodos.loc[gen_data.index, 'SG'] = gen_data['SG']
    return nodos
