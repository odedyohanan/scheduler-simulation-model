#!/bin/zsh
# Stop every long job of this project (data generation, training, evaluation), including pool workers.
pkill -f "experiments/(generate_data|train|evaluate|rollout).py" && echo "stopped"
sleep 1
pgrep -fl "experiments/(generate_data|train|evaluate|rollout).py" || echo "nothing running"
