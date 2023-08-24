import numpy as np
import pandas as pd


### BUS DATA

def bus_data_df(archivo):
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
    columnas_numericas = ['BASKV', 'I', 'AREA', 'ZONE', 'VM', 'VA', 'NVHI', 'NVLO', 'EVHI', 'EVLO']
    bus_data[columnas_numericas] = bus_data[columnas_numericas].apply(pd.to_numeric)

    # Muestra el DataFrame resultante
    return bus_data



### LOAD DATA

def load_data_df(archivo):
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
    columnas_numericas = ['PL', 'QL', 'IP', 'IQ', 'YP','YQ','SCALE','INTRPT',  'DGENP', 'DGENQ', 'DGENF']
    load_data[columnas_numericas] = load_data[columnas_numericas].apply(pd.to_numeric)

    # Muestra el DataFrame resultante
    return load_data


### SHUNT DATA

def shunt_data_df(archivo):
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


### BRANCH DATA

def branch_data_df(archivo):
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
    columnas_numericas = ['R', 'X', 'B']
    branch_data[columnas_numericas] = branch_data[columnas_numericas].apply(pd.to_numeric)

    branch_data = branch_data[['I', 'J', 'R', 'X', 'B']]

    # Muestra el DataFrame resultante
    return branch_data