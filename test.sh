#!/bin/sh

python test.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Ti64_orthogonal_sectioning' --model 'edsr'  --save 'edsr_l1_ti64' --resume -1  --model_to_load 'edsr_l1_ti64' --test_dataset_type 'Test' --test_only  --dist_type 'l1' 
