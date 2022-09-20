# EBSD-Superresolution: Adaptable Physics-Based Super-Resolution for Electron Backscatter Diffraction Maps
[Devendra K. Jangid*](), [Neal R. Brodnik*](), [Michael G. Goebel](), [Amil Khan](), [SaiSidharth Majeti](), [McLean P. Echlin](), [Samantha H. Daly](), [Tresa M. Pollock](), [B.S. Manjunath]()

[* equal contirbution]

[![paper]()

<hr />
> **Abstract:** *In computer vision, single image super-resolution (SISR) has been extensively explored using convolutional neural networks (CNNs) on optical images, but images outside this domain, such as those from scientific experiments, are not well investigated. Experimental data is often gathered using non-optical methods, which alters the metrics for image quality. One such example is electron backscatter diffraction (EBSD), a materials characterization technique that maps crystal arrangement in solid materials, which provides insight into processing, structure, and property relationships.  We present a broadly adaptable approach for applying state-of-art SISR networks to generate super-resolved EBSD orientation maps.  This approach includes quaternion-based orientation recognition, loss functions that consider rotational effects and crystallographic symmetry, and an inference pipeline to convert network output into established visualization formats for EBSD maps. The ability to generate physically accurate, high-resolution EBSD maps with super-resolution enables high throughput characterization and broadens the capture capabilities for three-dimensional experimental EBSD datasets.*

<hr />

## EBSD Framework
<img src = figures/>


## Installation
Step 1: Clone repo  

      git clone "https://github.com/UCSB-VRL/EBSD-Superresolution.git"
      
Step 2: Create Virtual environment

      virtualenv -p /usr/bin/python3.6 ebsdr_sr_venv(name of virtual environment)

Step 3: Activate Virtual environment

      source ebsd_sr_venv/bin/activate
      
Step 4: Download Dependencies

      pip install -r requirements.txt
      
Step 5: Install gradual warmup scheduler. Go to pytorch-gradual-warmup-lr folder

       python setup.py install
       

## Training 
Run
```
./train.sh
```
Define the following parameters to train network
   
* --input_dir "Directory Path to Datasets"
* --hr_data_dir "Directory Path to High Resolution Datasets"
* --val_lr_data_dir "Directory Path to Low Res Val Datasets"
* --val_hr_data_dir "Directory Path to High Res Val Datasets"
* --model "Network Architecture"
* --n_resblocks "Number of Residual Blocks"
* --n_resgroups "Number of Residual Groups"
* --n_colors "Number of Channel (for quaternion, it is 4)"
* --save "Folder name to save weights, loss curves and logs"
* --loss "Type of Loss for e.g. (misorientation.py has all combinations L1, L2 , tanh activation and rotational distance)"
* --dist_type "Which dist type for Loss (L1/L2/riot_dist_approx)"
   
Important parameters in argparser.py 
   
* --syms_req "It tells whether you want to use symmetry or not during Loss calculation"
* --GPU_Ids "Default gpu number is zero"
* --patch_size "Size of Patch During Training"
* --act "Activation Function in Network"
* --save_model_freq "How frequently do you want to save models"
* --act_loss "Whether you want to use tanh activation or not"
      



## Evaluation
Run
```
./test.sh
```
use "--test_only" flag as True

Inference model will be available on BisQue (https://bisque2.ece.ucsb.edu/client_service/) as module. 

## Results

## Acknowledgements

## Citation

## Contact
Should you have any question, please contact dkjangid@ucsb.edu or nbrodnik@ucsb.edu
       
