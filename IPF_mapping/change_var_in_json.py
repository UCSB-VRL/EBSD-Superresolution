import json
from argparser import Argparser

args = Argparser().args

f = open('pipeline.json')

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


 
print(f'Tupel: {Tupel}')
npy_file_dir = f'{args.fpath}/{args.dataset_type}_{args.model_name}'
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
