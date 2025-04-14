#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
# Remove progessive patch
python main.py --optimizer 'SGD' --model 'edsr' --lr 1e-6 --dist_type 'minimum_angle_transformation' --epochs 2000 --batch_size 4 --n_feats 32 --syms_req --syms 'fcc' --val_freq 1 --root_dir '/media/hdd3/jmgiorgi/EBSD-ref-symm' --save 'Open_718_MAT_2025' --GPU_ID 0 --n_GPUs 1 --patch_size 64  --input_dir '/media/hdd3/jmgiorgi/fz_reduced/Open_718' --hr_data_dir 'Train/HR_Images' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Blocks'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Blocks' 
