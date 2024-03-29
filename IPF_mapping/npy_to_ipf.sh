#!/bin/bash

modelname="edsr_l1_ti64"
filetype="SR"
datasettype=("Test")
material="Ti64"
sect=("X_Block" "Y_Block" "Z_Block")

# path of Dream3D software
dream3d_path="/home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping/DREAM3D-6.5.141-Linux-x86_64/bin"

#path of json file for Dream3D pipeline
json_path="/home/dkjangid/Material_Project/EBSD_Superresolution/NPJ_Repo/EBSD-Superresolution/IPF_mapping/pipeline.json"

# path of IPF mapping code
home_path="/home/dkjangid/Material_Project/EBSD_Superresolution/NPJ_Repo/EBSD-Superresolution/IPF_mapping"

# path where numpy files (output of models) are saved
file_path="/home/dkjangid/Material_Project/EBSD_Superresolution/NPJ_Repo/EBSD-Superresolution/experiments/saved_weights/edsr_l1_ti64/results"

#path of your source dream3d file as refernece
sourcename="/data/dkjangid/superresolution/Material_Dataset/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d"


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


