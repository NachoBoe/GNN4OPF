# from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
# import glob
import os
import argparse
import json

# path = "/home/iboero/Documents/GNN4OPF/Entrenar/runs/30/FCNN_global/"

def find_best_run(path, k=10):
    '''
    Encuentra los mejores k entrenamientos en un directorio, a partir de los info.json
    '''
    entrenamiento = []
    val_loss = []
    feasibility = []
    voltage_setpoint = []

    for file in os.listdir(path):
        with open(os.path.join(path, file,'best_model_info.json'), 'r') as f:
            datos = json.load(f)
        val_loss.append(datos['val_loss'])
        feasibility.append(datos['feasibility_metric'])
        voltage_setpoint.append(datos['voltaje_setpoint_metric'])
        entrenamiento.append(datos['model_name'].split('/')[-1])


    indices_ordenados = sorted(range(len(voltage_setpoint)), key=lambda i: voltage_setpoint[i], reverse=False)
    best_values = indices_ordenados[:k]
    k_best_losses = [val_loss[i] for i in best_values]
    k_best_feasibilities = [feasibility[i] for i in best_values]
    k_best_voltage_setpoints = [voltage_setpoint[i] for i in best_values]
    k_best_trains = [entrenamiento[i] for i in best_values]

    return k_best_trains, k_best_losses, k_best_feasibilities, k_best_voltage_setpoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encuentra los mejores k entrenamientos en un directorio, a partir de los events.out.tfevents')
    parser.add_argument('folder_path', type=str, help='Carpeta con todos los entrenamientos')
    parser.add_argument('-k', type=int, default=10, help='Número de los mejores entrenamientos a encontrar (por defecto: 10)')
    args = parser.parse_args()

    k_best_trains = find_best_run(args.folder_path, args.k)
    print(k_best_trains)