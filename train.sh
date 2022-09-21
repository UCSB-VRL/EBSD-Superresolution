#!/bin/sh

#cd pytorch-gradual-warmup-lr

#python setup.py install

#cd ../

python main.py --input_dir 'data/dkjangid/superresolution/Material_Dataset/Titanium_Dataset/fz_reduced/Titanium_all_data' --hr_data_dir 'Train/HR_Images/preprocessed_imgs_all_blocks' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Block'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Block' --model 'han' --n_resblocks 20 --n_resgroups 10 --n_feats 128 --n_colors 4 --save 'han_rotdist_symm_titanium_all_blcks_dataset_debug'  --loss '1*MisOrientation' --dist_type 'rot_dist_approx' --patch_size 64 --batch_size 128 --syms_req 
