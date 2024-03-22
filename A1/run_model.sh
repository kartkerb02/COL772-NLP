#!/bin/bash

# Set the task (train/test)
task=$1

# Paths to data and save directories
dir1=$2
dir2=$3

# Train the model
if [ "$task" == "train" ]; then
    python3 train_model.py "$dir1" "$dir2"
    echo "Model training complete. Saved in: $dir2"
fi

# Test the model
if [ "$task" == "test" ]; then
    python3 predict_model.py "$dir1" "$dir2" "output.txt"
    echo "Inference complete. Predictions saved in: output.txt"
fi

# sample usage
# bash run_model.sh train <path_to_data_json> <path_to_save>
# bash run_model.sh test <path_to_save> <path_to_test_json> output.txt

# bash run_model.sh train ./data/ ./
# bash run_model.sh test ./ ./data/valid_new.json output.txt