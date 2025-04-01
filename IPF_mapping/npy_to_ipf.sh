#!/bin/bash

# modelname"edsr_rot_dist_approx_prog_pa"
modelname='Ti_han_minimum_angle_transformation_2000_epochs'
filetype="SR"
datasettype=("Test")
material="Ti64"
# sect=("X_Block")
sect=("X_Block" "Y_Block" "Z_Block")

# path of Dream3D software
dream3d_path="/media/hdd3/jmgiorgi/DREAM3D-6.5.171-Linux-x86_64/bin"

#path of json file for Dream3D pipeline
json_path="/home/jmgiorgi/EBSD-ref-symm/EBSD-Superresolution/IPF_mapping/pipeline.json"

# path of IPF mapping code
home_path="/home/jmgiorgi/EBSD-ref-symm/EBSD-Superresolution/IPF_mapping"

# path where numpy files (output of models) are saved
file_path="/media/hdd3/jmgiorgi/EBSD-ref-symm/experiments/saved_weights/Ti_han_minimum_angle_transformation_2000_epochs/results"
# file_path="/media/hdd3/jmgiorgi/SLERP/Open_718/X4"
# file_path="/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/HR_Images"

#path of your source dream3d file as refernece
sourcename="/media/hdd3/jmgiorgi/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d"

for s in ${sect[@]}; do
   
    echo "Running Numpy to Dream3D"
    python npy_to_dream3d.py --fpath $file_path --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s --d3_source $sourcename
    
    echo "Changing Variable in JSON "

    python change_var_in_json.py --fpath $file_path  --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s
    
    echo "Running Dream 3D Pipeline"
        
    cd $dream3d_path
    
    ./PipelineRunner -p $json_path

    echo "Running Dream 3D Pipeline"
    
    cd $home_path
  
    python dream3d_to_rgb.py --fpath $file_path --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s

done

