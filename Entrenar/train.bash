#!/bin/bash

# Define las listas de hiperparámetros a probar
batch_norm_options=(True False)
device_options=('cuda')
batch_size_options=(64 128)
lr_options=(1e-3 1e-4)
red_options=('30' '118')
K_options=("[4,4,4]" "[3,3,3]" "[5,5,5]")
layers_options=("[3,64,64,1]" "[3,256,256,1]" "[3,512,512,1]")

# Tomar model_option de la línea de comandos
model_option=$1

# Calcula el total de combinaciones posibles
total_combinations=$((${#batch_norm_options[@]} * 1 * ${#device_options[@]} * ${#batch_size_options[@]} * ${#lr_options[@]} * ${#red_options[@]} * ${#K_options[@]} * ${#layers_options[@]}))

# Contador para correr procesos en paralelo hasta un máximo de 1 para depuración
max_jobs=1
jobs=0
count=0

generate_yaml_config() {
    bn=$1
    m=$2
    dev=$3
    bs=$4
    r=$5
    k="${6//[\[\]]/}"
    ly="${7//[\[\]]/}"
    lr=$8

    config_name="config_${m}_red${r}_bs${bs}_lr${lr//.}"
    filename="configs_reactiva/${config_name}.yaml"

    cat <<EOF > "$filename"
outdir: runs

model:
  batch_norm: $bn
  model: $m
  layers: [$ly]
  K: [$k]

data:
  data_path: ../data/data_reactiva/uniforme_07_13_red$r
  red: '$r'
  normalize_X: True
  normalize_Y: False

training:
  device: '$dev'
  batch_size: $bs
  lr: $lr
  num_epochs: 500
  early_stopping: 30
  betas: [0.9, 0.999]
  weight_decay: 0
  seed: 42
  initial_metric_epoch: 10
  metric_frec: 30
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
  yaml_file=$(generate_yaml_config $batch_norm $model_option $device $batch_size $red $K $layers $lr)

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

# Esperar a que todos los trabajos terminen
wait

echo "Grid search completado."