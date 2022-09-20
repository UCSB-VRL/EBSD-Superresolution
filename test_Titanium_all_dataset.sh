#!/bin/sh

python test.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Titanium_all_data' --model 'han' --n_resblocks 20 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'han_l1_titanium_all_blcks_dataset'  --loss '1*MisOrientation' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'l1' 
