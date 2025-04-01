#!/bin/sh

python test.py --input_dir '/media/hdd3/jmgiorgi/Titanium_all_data' --model 'edsr'  --save 'Ti_edsr_rot_dist_approx' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'rot_dist_approx' 
