## ENTRADA

El archivo input.npy consiste en un array de Nx30x3.
    - N corresponde a la cantidad de datos.
    - 30 corresponde a la cantidad de buses
    - 3 corresponde a las features de entrada por nodo, que son:
        - Columna 0: Corresponde a la potencia activa neta consumida en el nodo
        - Columna 1: Corresponde a la potencia reactiva neta consumida en el nodo
        - Columna 2: Corresponde a la potencia activa neta generada en el nodo


## SALIDA

El archivo vm_pu_opt.npy consiste en un array de Nx30x1.
    - N corresponde a la cantidad de datos.
    - 30 corresponde a la cantidad de buses
    - 1 corresponde a las features de salida por nodo, que son:
        - Columna 0: El voltaje setpoint óptimo para el nodo generador obtenido a partir de resolver el OPF.
        Nota: El valor de esta columna es 0 para los buses sin generadores (buses PQ)