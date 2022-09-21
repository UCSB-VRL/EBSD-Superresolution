import h5py
import numpy as np
import glob
import os
from PIL import Image 
from argparser import Argparser

args = Argparser().args

#npy_file_dir = f'../../experiment/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

npy_file_dir = f'/data/dkjangid/Material_Projects/superresolution/AAAI22_after/saved_weights/{args.model_name}/results/{args.dataset_type}_{args.model_to_load}'

file_locs = sorted(glob.glob(f'{npy_file_dir}/{args.data}/*{args.section}*{args.file_type}*.npy'))

total_file = len(file_locs)

dream_3d_file = f'{npy_file_dir}/{args.data}/Dream3D/{args.section}_{args.file_type}.dream3d'

dream3d_file = h5py.File(f'{dream_3d_file}')


img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']

for i, file_loc in enumerate(file_locs):
   
    basename = os.path.basename(file_loc)
    filename = os.path.splitext(basename)[0]
   
    image = Image.fromarray(img[i,:,:,:], "RGB")
    image.save(f'{npy_file_dir}/{args.data}/Dream3D/{filename}.png')
