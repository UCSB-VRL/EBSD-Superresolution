#!/bin/sh

python test.py --input_dir '/media/hdd3/jmgiorgi/Titanium_all_data' --model 'han' --n_feats 32 --save 'Ti_han_minimum_angle_transformation_2000_epochs' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'minimum_angle_transformation' 
