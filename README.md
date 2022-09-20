# EBSD-Superresolution: Adaptable Physics-Based Super-Resolution for Electron Backscatter Diffraction Maps
[Devendra K. Jangid*](), [Neal R. Brodnik*](), [Michael G. Goebel](), [Amil Khan](), [SaiSidharth Majeti](), [McLean P. Echlin](), [Samantha H. Daly](), [Tresa M. Pollock](), [B.S. Manjunath]()

[* equal contirbution]

[![paper]()

<hr />
> **Abstract:** *In computer vision, single image super-resolution (SISR) has been extensively explored using convolutional neural networks (CNNs) on optical images, but images outside this domain, such as those from scientific experiments, are not well investigated. Experimental data is often gathered using non-optical methods, which alters the metrics for image quality. One such example is electron backscatter diffraction (EBSD), a materials characterization technique that maps crystal arrangement in solid materials, which provides insight into processing, structure, and property relationships.  We present a broadly adaptable approach for applying state-of-art SISR networks to generate super-resolved EBSD orientation maps.  This approach includes quaternion-based orientation recognition, loss functions that consider rotational effects and crystallographic symmetry, and an inference pipeline to convert network output into established visualization formats for EBSD maps. The ability to generate physically accurate, high-resolution EBSD maps with super-resolution enables high throughput characterization and broadens the capture capabilities for three-dimensional experimental EBSD datasets.*

<hr />

## EBSD Framework
<img src = >


## Installation
Step 1: Clone repo  

      git clone ""
      
Step 2: Create Virtual environment

      virtualenv -p /usr/bin/python3.6 ebsdr_sr_venv(name of virtual environment)

Step 3: Activate Virtual environment

      source ebsd_sr_venv/bin/activate
      
Step 4: Download Dependencies

      pip install -r requirements.txt
      
Step 5: Install gradual warmup scheduler. Go to pytorch-gradual-warmup-lr folder

       python setup.py install
       

## Training 

## Evaluation
       
