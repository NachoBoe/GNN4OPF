from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import glob
import os
import argparse
import json

# path = "/home/iboero/Documents/GNN4OPF/Entrenar/runs/30/FCNN_global/"

def find_best_run_from_tfevents(path, k=10):
    '''
    Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents
    '''
    last = []
    entrenamiento = []

    for file in os.listdir(path):
        evento = glob.glob(path + file + '/' + "events.out.tfevents*")[0]
        event_loader = EventFileLoader(evento)
        events = event_loader.Load()
        for event in events:
            evento_final = event
        try:
            last.append(evento_final.summary.value[0].tensor.float_val[0])
            entrenamiento.append(file)
        except:
            pass

    indices_ordenados = sorted(range(len(last)), key=lambda i: last[i], reverse=False)
    best_values = indices_ordenados[:k]
    k_best_trains = [entrenamiento[i] for i in best_values]

    return k_best_trains

def find_best_run_from_tfevents(path, k=10):
    '''
    Encuentra los mejores k entrenamientos en un directorio, a partir de los info.json
    '''
    last = []
    entrenamiento = []

    for file in os.listdir(path):
        with open('datos.json', 'r') as f:
            datos = json.load(f)

        for event in events:
            evento_final = event
        try:
            last.append(evento_final.summary.value[0].tensor.float_val[0])
            entrenamiento.append(file)
        except:
            pass

    indices_ordenados = sorted(range(len(last)), key=lambda i: last[i], reverse=False)
    best_values = indices_ordenados[:k]
    k_best_trains = [entrenamiento[i] for i in best_values]

    return k_best_trains


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents')
    parser.add_argument('folder_path', type=str, help='Carpeta con todos los entrenamientos')
    parser.add_argument('-k', type=int, default=10, help='Número de los mejores entrenamientos a encontrar (por defecto: 10)')
    args = parser.parse_args()

    k_best_trains = find_best_run(args.folder_path, args.k)
    print(k_best_trains)