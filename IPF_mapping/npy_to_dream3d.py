import h5py
import numpy as np
import dream3d_import as d3
import glob
import os
from argparser import Argparser

args = Argparser().args

# npy_file_dir = f'{args.fpath}/{args.dataset_type}_model_best'

import pdb; pdb.set_trace()
 
npy_file_dir = f''
file_locs = []
if (args.file_type == "hr" or args.file_type == "lr" or args.file_type == "HR" or args.file_type == "LR"):
    npy_file_dir = f'{args.fpath}'
    file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))
elif (args.file_type == "SR" or args.file_type == "sr"):
    pdb.set_trace()
    npy_file_dir = f'{args.fpath}/{args.dataset_type}_model_best'
    file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))
else: # file_type will equal SLERPED in this case
    npy_file_dir = f'{args.fpath}'
    file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*SLERPED*.npy'))
    
    
total_file = len(file_locs)

arr_list = []
for file_loc in file_locs:
    arr = np.load(file_loc)
    print(arr.shape)
    arr_list.append(arr)

loaded_npy = np.asarray(arr_list)
     
#d3_sourceName = '/data/dkjangid/superresolution/Material_Dataset/Ti64_DIC_Homo_and_Cubochoric_FZ.dream3d'
d3_sourceName=f'{args.d3_source}'


# The path for the output Dream3D file being written.  This is where you want to save the file you are making.

import pdb; pdb.set_trace()

save_path = f'{npy_file_dir}/Dream3D'

if not os.path.exists(f'{npy_file_dir}/Dream3D'):
    os.makedirs(f'{save_path}')

d3_outputName = f'{save_path}/{args.section}_{args.file_type}.dream3d'

d3source = h5py.File(d3_sourceName, 'r')

xdim,ydim,zdim,channeldepth = np.shape(loaded_npy)

phases = np.int32(np.ones((xdim,ydim,zdim)))

new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)

in_path = 'DataContainers/ImageDataContainer' 
out_path = 'DataContainers/ImageDataContainer'

import pdb; pdb.set_trace()
new_file = d3.copy_container(d3_sourceName, f'{in_path}/CellEnsembleData', d3_outputName, f'{out_path}/CellEnsembleData')

new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim), source_internal_geometry_path=f'{in_path}/_SIMPL_GEOMETRY', output_internal_geometry_path=f'{out_path}/_SIMPL_GEOMETRY')

new_file = d3.create_empty_container(d3_outputName, f'{out_path}/CellData', (xdim,ydim,zdim), 4)
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', loaded_npy, 'Quats')
# Changed this to directly input to Euler for adaptive filter!! #
# new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', loaded_npy, 'Euler')
new_file = d3.add_to_container(d3_outputName, f'{out_path}/CellData', phases, 'Phases')

# Close out source file to avoid weird memory errors.
d3source.close()
