import json
from argparser import Argparser

args = Argparser().args

f = open('pipeline.json')

#import pdb;pdb.set_trace()

field_dict = json.load(f)

data = args.data

if data == 'Ti64':
    if args.section == "X_Block":
        Tupel = [471, 142, 18]
    elif args.section == "Y_Block":
        Tupel = [471, 346, 7]
    elif args.section == "Z_Block":
        Tupel = [142, 346, 24]

elif data == 'Ti7_1Percent':
    if args.section == "X_Block":
        Tupel = [434, 551, 9]
    elif args.section == "Y_Block":
        Tupel = [220, 551, 37]
    elif args.section == "Z_Block":
        Tupel = [220, 434, 47]

elif data == 'Ti7_3Percent':
    if args.section == "X_Block":
        Tupel = [510, 652, 18]
    elif args.section == "Y_Block":
        Tupel = [213, 652, 43]
    elif args.section == "Z_Block":
        Tupel = [213, 510, 56]

elif data == 'Open_718':
    if args.section == 'x_block':
        Tupel = [301, 390, 191]
    if args.section == 'y_block':
        Tupel = [301, 191, 390]
    if args.section == 'z_block':
        Tupel = [390, 191, 301] 

 
print(f'Tupel: {Tupel}')

filedir = f''
if (args.file_type == "HR" or args.file_type == "LR" or args.file_type == "hr" or args.file_type == "lr"):
    filedir = f'{args.fpath}/Dream3D/{args.section}_{args.file_type}.dream3d'
elif (args.file_type == "SR" or args.file_type == "sr"):
    npy_file_dir = f'{args.fpath}/{args.dataset_type}_model_best'
    filedir = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'
else: # file_type will equal SLERPED in this case
    npy_file_dir = f'{args.fpath}'
    filedir = f'{npy_file_dir}/Dream3D/{args.section}_{args.file_type}.dream3d'

#field_dict["0"]["InputFile"] = f'/home/dkjangid/Material_Project/EBSD_Superresolution/{filedir}'

field_dict["0"]["InputFile"] = f'{filedir}'

#field_dict["3"]["OutputFile"] = f'/home/dkjangid/Material_Project/EBSD_Superresolution/{filedir}'
field_dict["3"]["OutputFile"] = f'{filedir}'


field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][0]["Tuple Dimensions"]= Tupel

field_dict["0"]["InputFileDataContainerArrayProxy"]["Data Containers"][0]["Attribute Matricies"][0]["Data Arrays"][1]["Tuple Dimensions"]= Tupel



outfile = open("pipeline.json", "w")

json.dump(field_dict, outfile, indent=4)

outfile.close()
