# EBSD-Superresolution: Adaptable Physics-Based Super-Resolution for Electron Backscatter Diffraction Maps
[Devendra K. Jangid*](https://www.sites.google.com/view/dkj910/), [Neal R. Brodnik*](https://scholar.google.com/citations?user=3dAoFJkAAAAJ&hl=en), [Michael G. Goebel](https://scholar.google.com/citations?user=FwMJrygAAAAJ&hl=en), [Amil Khan](https://scholar.google.com/citations?user=r6jNH5UAAAAJ&hl=en), [SaiSidharth Majeti](), [McLean P. Echlin](https://scholar.google.com/citations?user=fxN2OsUAAAAJ&hl=en), [Samantha H. Daly](https://scholar.google.com/citations?user=3whYx4UAAAAJ&hl=en), [Tresa M. Pollock](https://materials.ucsb.edu/people/faculty/tresa-pollock), [B.S. Manjunath](https://scholar.google.com/citations?user=wRYM4qgAAAAJ&hl=en)

[* equal contirbution]

[paper]()

<hr />

> **Abstract:** *In computer vision, single image super-resolution (SISR) has been extensively explored using convolutional neural networks (CNNs) on optical images, but images outside this domain, such as those from scientific experiments, are not well investigated. Experimental data is often gathered using non-optical methods, which alters the metrics for image quality. One such example is electron backscatter diffraction (EBSD), a materials characterization technique that maps crystal arrangement in solid materials, which provides insight into processing, structure, and property relationships.  We present a broadly adaptable approach for applying state-of-art SISR networks to generate super-resolved EBSD orientation maps.  This approach includes quaternion-based orientation recognition, loss functions that consider rotational effects and crystallographic symmetry, and an inference pipeline to convert network output into established visualization formats for EBSD maps. The ability to generate physically accurate, high-resolution EBSD maps with super-resolution enables high throughput characterization and broadens the capture capabilities for three-dimensional experimental EBSD datasets.*
<hr />

## EBSD Framework
<img src = "images/EBSD_SR_train_val_loss_comp_lg.png">


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
<table>
      <tr>
          <th align="center">Loss</th>
          <th align="center">dist_type</th>
           <th align="center">syms_req</th>  
      </tr>
       <tr>
          <td align="center">L1</td>
          <td align="center">L1</td>
          <td align="center">False</td>  
      </tr>
       <tr>
          <td align="center">L1 with symmetry</td>
          <td align="center">L1</td>
          <td align="center">True</td>  
      </tr>
        <tr>
          <td align="center">Rotational distance approximation with symmetry</td>
          <td align="center">rot_dist_approx</td>
          <td align="center">True</td>  
      </tr>
<table>


Define the following parameters to train network
   
* ```--input_dir```: "Directory Path to Datasets"
* ```--hr_data_dir```: "Path to High Resolution EBSD Maps relative to input_dir"
* ```--val_lr_data_dir```: "Path to Low Resolution EBSD Val Datasets"
* ```--val_hr_data_dir```: "Path to High Resolution EBSD Val Datasets"
* ```--model```: "Choose one of network architectures from edsr, rfdn, san, han"
* ```--save```: "Folder name to save weights, loss curves and logs"
   
Important parameters in argparser.py 
   
* ```--syms_req```: "It tells whether you want to use symmetry or not during Loss calculation"
* ```--patch_size```: "Size of Patch During Training"
* ```--act```: "Activation Function in Network"
* ```--save_model_freq```: "How frequently do you want to save models"

## Evaluation
Run
```
./test.sh
```
use ```--test_only``` flag as True

Inference model will be available on [BisQue](https://bisque2.ece.ucsb.edu/client_service/) as module. 

## Results
<img src = "images/QualitativeResults.png">

## Datasets
Material datasets will be available by request at discretion of authors. 

## Acknowledgements

## Citation

## Contact
Should you have any question, please contact dkjangid@ucsb.edu or nbrodnik@ucsb.edu
       
