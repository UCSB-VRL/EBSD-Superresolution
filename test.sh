#!/bin/sh

python test.py --input_dir '/media/hdd3/jmgiorgi/fz_reduced/Open_718' --model 'edsr' --n_feats 32 --save 'Open_718_MAT_2025_lr1e-2' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'minimum_angle_transformation' 
