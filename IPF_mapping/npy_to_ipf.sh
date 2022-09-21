#!/bin/bash


modelname="han_rotdist_symm_titanium_dataset"
modeltoload="model_best"
filetype="SR"

#datasettype=("Val" "Test")
datasettype=("Test")

#material=("Ti7_3Percent")
materials=("Ti64" "Ti7_1Percent" "Ti7_3Percent")

sect=("X_Block" "Y_Block" "Z_Block")
#sect=("X_Block" )


for material in ${materials[@]};do

	for d_type in ${datasettype[@]}; do
		for s in ${sect[@]}; do
			echo "$d_type  $s" 

			echo "Running Numpy to Dream3D"
			python npy_to_dream3d.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
			
			echo "Changing Variable in JSON "

			python change_var_in_json.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
			

	  
			echo "Running Dream 3D Pipeline"

			#path to Dream3D program
			cd /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping/DREAM3D-6.5.141-Linux-x86_64/bin 
			
			
			./PipelineRunner -p /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping/pipeline.json

			echo "Running Dream 3D Pipeline"
			
			cd /home/dkjangid/Material_Project/EBSD_Superresolution/github_version/IPF_mapping 
			python dream3d_to_rgb.py --data $material --model_name $modelname --model_to_load $modeltoload --file_type $filetype --dataset_type $d_type --section $s
		


		done
	done

done
