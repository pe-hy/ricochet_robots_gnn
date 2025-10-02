#!/bin/bash

# Run all tasks sequentially
tasks=("basic" "threshold" "unreachable" "regression" "best_component")

for task in "${tasks[@]}"; do
    echo "=========================================="
    echo "Starting training for task: $task"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=1 python train.py data.task=$task

    if [ $? -ne 0 ]; then
        echo "Error: Training failed for task $task"
        exit 1
    fi

    echo ""
    echo "Completed task: $task"
    echo ""
done

echo "=========================================="
echo "All tasks completed successfully!"
echo "=========================================="
