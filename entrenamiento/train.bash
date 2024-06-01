#!/bin/bash

# Define las listas de hiperparámetros a probar
batch_norm_options=(True)
device_options=('cuda')
batch_size_options=(64 128)
lr_options=(1e-4 1e-5)
red_options=('118')
K_options=("[2,2,2,2,2,2]")
layers_options=("[3,512,2048,2048,2048,512,4]")
dual_coef_real_options=(1 0.1 0.01)
dual_coef_imag_options=(1 0.1 0.01)

# Tomar model_option de la línea de comandos
model_option=$1

# Calcula el total de combinaciones posibles
total_combinations=$((${#batch_norm_options[@]} * 1 * ${#device_options[@]} * ${#batch_size_options[@]} * ${#lr_options[@]} * ${#red_options[@]} * ${#K_options[@]} * ${#layers_options[@]} * ${#dual_coef_real_options[@]} * ${#dual_coef_imag_options[@]}))

# Contador para correr procesos en paralelo hasta un máximo de 1 para depuración
max_jobs=1
jobs=0
count=0

generate_yaml_config() {
    bn=$1
    dev=$2
    bs=$3
    r=$4
    k="${5//[\[\]]/}"
    ly="${6//[\[\]]/}"
    lr=$7
    dcr=$8
    dci=$9

    config_name="config_bash"
    filename="configs/${config_name}.yaml"

    cat <<EOF > "$filename"
outdir: ../resultados/runs

model:
  layers: [$ly] # La ultima es out_dim
  K: [$k]

data:
  data_path: /home/iboero/Tesis/unsupervised_ieee/GNN4OPF/data
  red: '$r' # 30 o 118
  red_path: None
  normalize_X: False

training:
  device: '$dev'
  batch_size: $bs
  lr: $lr
  num_epochs: 500
  early_stopping: 30
  betas: [0.9, 0.999]
  weight_decay: 0
  seed: 42
  metric_frec: 30
  initial_metric_epoch: 0
  batch_norm: $bn # Booleano
  dual_coefs: [$dcr, $dci, 0] # dual_acflow_real, dual_acflow_imag, dual_lines 
EOF

    echo "$filename"
}

for red in "${red_options[@]}"; do
for batch_norm in "${batch_norm_options[@]}"; do
for device in "${device_options[@]}"; do
for batch_size in "${batch_size_options[@]}"; do
for lr in "${lr_options[@]}"; do
for K in "${K_options[@]}"; do
for layers in "${layers_options[@]}"; do
for dual_coef_real in "${dual_coef_real_options[@]}"; do
for dual_coef_imag in "${dual_coef_imag_options[@]}"; do
  yaml_file=$(generate_yaml_config $batch_norm $device $batch_size $red $K $layers $lr $dual_coef_real $dual_coef_imag)
  
  # Ejecutar en background
  python train.py --cfg "$yaml_file" &
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
done
done
done
done
done
done
done

# Esperar a que todos los trabajos terminen
wait

echo "Grid search completado."