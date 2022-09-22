import h5py
import numpy as np
import dream3d_import as d3
import glob
import os
from argparser import Argparser

args = Argparser().args

npy_file_dir = f'{args.fpath}/{args.dataset_type}_{args.model_name}'

file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))

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

save_path = f'{npy_file_dir}/Dream3D'

if not os.path.exists(f'{npy_file_dir}/Dream3D'):
    os.makedirs(f'{save_path}')

d3_outputName = f'{save_path}/{args.section}_{args.file_type}.dream3d'

d3source = h5py.File(d3_sourceName, 'r')

xdim,ydim,zdim,channeldepth = np.shape(loaded_npy)

phases = np.int32(np.ones((xdim,ydim,zdim)))

new_file = d3.create_dream3d_file(d3_sourceName, d3_outputName)



new_file = d3.copy_container(d3_sourceName, 'DataContainers/ImageDataContainer/CellEnsembleData', d3_outputName,
                          'DataContainers/ImageDataContainer/CellEnsembleData')


new_file = d3.create_geometry_container_from_source(d3_sourceName, d3_outputName, dimensions=(xdim,ydim,zdim),
                            source_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY',
                            output_internal_geometry_path='DataContainers/ImageDataContainer/_SIMPL_GEOMETRY')

new_file = d3.create_empty_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', (xdim,ydim,zdim), 3)
new_file = d3.add_to_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', loaded_npy, 'Quats')
new_file = d3.add_to_container(d3_outputName, 'DataContainers/ImageDataContainer/CellData', phases, 'Phases')

# Close out source file to avoid weird memory errors.
d3source.close()
