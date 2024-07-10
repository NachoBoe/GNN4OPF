#!/bin/bash

# Define las listas de hiperparámetros a probar
red_options=('30' '118')
K_options=('[2,2,2,2,2,2,2,2]' '[3,3,3,3,3,3,3,3]')


# Tomar model_option de la línea de comandos
model_option=$1

# Contador para correr procesos en paralelo hasta un máximo de 1 para depuración
max_jobs=1
jobs=0
count=0

generate_yaml_config() {
    r=$1
    k="${2//[\[\]]/}"

    config_name="config_bash"
    filename="configs/${config_name}.yaml"

    cat <<EOF > "$filename"
outdir: ../resultados/runs

model:
  layers: [3,512, 2048, 2048,2048,2048, 2048, 512, 4] # La ultima es out_dim
  K: [2,2,2,2,2,2,2,2]

data:
  data_path: /home/iboero/Tesis/unsupervised_ieee/GNN4OPF/data
  red: '$r' # 30 o 118
  red_path: None
  normalize_X: False

training:
  device: 'cuda'
  batch_size: 0
  lr: 0
  num_epochs: 500
  early_stopping: 30
  betas: [0.9, 0.999]
  weight_decay: 0
  seed: 42
  metric_frec: 30
  initial_metric_epoch: 0
  batch_norm: True
  use_edge_weights: True
  dual_coefs: [0, 0, 0] # dual_acflow_real, dual_acflow_imag, dual_lines 
EOF

    echo "$filename"
}

for red in "${red_options[@]}"; do
for K in "${K_options[@]}"; do
  yaml_file=$(generate_yaml_config $red $K)
  
  # Ejecutar en background
  python train_optuna.py --cfg "$yaml_file" &
  let jobs+=1
  let count+=1

  # Imprimir el progreso
  echo "Starting job $count/$total_combinations with config $yaml_file"

  # Control de concurrencia
  if [ "$jobs" -ge $max_jobs ]; then
    wait
    jobs=0
  fi

done
done


# Esperar a que todos los trabajos terminen
wait

echo "Grid search completado."