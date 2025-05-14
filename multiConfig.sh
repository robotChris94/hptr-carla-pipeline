#!/bin/bash
# multi_config.sh
# 批次執行多組模型與路徑設定

MODEL_LIST=(
  "models/hptr_v1.pt"
  "models/hptr_v2.pt"
)

CONFIG_LIST=(
  "configs/route_town05.json"
  "configs/route_town03.json"
  "configs/route_town06.json"
)

REPEAT=20

for model in "${MODEL_LIST[@]}"; do
  for config in "${CONFIG_LIST[@]}"; do
    name=$(basename "$model" .pt)
    town=$(basename "$config" .json)
    logname="logs/${name}_${town}.json"
    echo "Running $model on $config -> $logname"
    python3 experiment_runner.py \
      --model_path "$model" \
      --config_path "$config" \
      --log "$logname" \
      --repeat $REPEAT
  done
done
