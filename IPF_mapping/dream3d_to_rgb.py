import h5py
import numpy as np
import glob
import os
from PIL import Image 
from argparser import Argparser

args = Argparser().args

import pdb; pdb.set_trace()



file_locs = []
dream_3d_file = f''
npy_file_dir = f''

if (args.file_type == "HR" or args.file_type == "LR" or args.file_type == "hr" or args.file_type == "lr"):
    npy_file_dir = f'{args.fpath}'
    file_locs = sorted(glob.glob(f'{args.fpath}/*{args.section}*.npy'))
    dream_3d_file = f'{args.fpath}/Dream3D/{args.section}_{args.file_type}.dream3d'
elif (args.file_type == "SR" or args.file_type == "sr"):
    npy_file_dir = f'{args.fpath}/{args.dataset_type}_model_best'
    file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))
    dream_3d_file = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'
else: # file_type will equal SLERPED in this case
    npy_file_dir = f'{args.fpath}'
    file_locs = sorted(glob.glob(f'{npy_file_dir}/*{args.section}*{args.file_type}*.npy'))
    dream_3d_file = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'

import pdb; pdb.set_trace()

total_file = len(file_locs)

# dream_3d_file = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'

dream3d_file = h5py.File(f'{dream_3d_file}')


img = dream3d_file['DataContainers']['ImageDataContainer']['CellData']['IPFColor']

for i, file_loc in enumerate(file_locs):
   
    basename = os.path.basename(file_loc)
    filename = os.path.splitext(basename)[0]
   
    image = Image.fromarray(img[i,:,:,:], "RGB")
    image.save(f'{npy_file_dir}/Dream3D/{filename}.png')
