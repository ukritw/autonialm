#!/bin/bash

# REDD AUTO House 1 - optimize for MAE
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance fridge --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "mean_absolute_error" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance light --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "mean_absolute_error" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance sockets --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "mean_absolute_error" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance "washer dryer" --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "mean_absolute_error" --max_evals 200

# REDD AUTO House 1 - optimize for accuracy_score
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance fridge --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "accuracy_score" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance light --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "accuracy_score" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance sockets --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "accuracy_score" --max_evals 200
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance "washer dryer" --sampling_rate 20 --epochs 2000 --patience 15 --metrics_to_optimize "accuracy_score" --max_evals 200

# REDD QUICK House 1 - optimize for MAE
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance fridge --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "mean_absolute_error" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance light --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "mean_absolute_error" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance sockets --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "mean_absolute_error" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance "washer dryer" --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "mean_absolute_error" --max_evals 50

# REDD QUICK House 1 - optimize for accuracy_score
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance fridge --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "accuracy_score" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance light --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "accuracy_score" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance sockets --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "accuracy_score" --max_evals 50
python automl_hyperopt.py --datapath /mnt/data/datasets/wattanavaekin/REDD/redd.h5 --train_building 1 --train_end 2011-05-14 --test_building 1 --test_start 2011-05-14 --appliance "washer dryer" --sampling_rate 20 --epochs 500 --patience 5 --metrics_to_optimize "accuracy_score" --max_evals 50
