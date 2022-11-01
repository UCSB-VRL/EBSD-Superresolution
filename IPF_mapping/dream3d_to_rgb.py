import h5py
import numpy as np
import glob
import os
from PIL import Image 
from argparser import Argparser

args = Argparser().args

npy_file_dir = f'{args.fpath}/{args.dataset_type}_{args.model_name}'

file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))

total_file = len(file_locs)

dream_3d_file = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'

dream3d_file = h5py.File(f'{dream_3d_file}')


img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']

for i, file_loc in enumerate(file_locs):
   
    basename = os.path.basename(file_loc)
    filename = os.path.splitext(basename)[0]
   
    image = Image.fromarray(img[i,:,:,:], "RGB")
    image.save(f'{npy_file_dir}/Dream3D/{filename}.png')
