# IPF Mapping

This code convert numpy files to IPF mapping, which is a standard method to visualize EBSD microstructure. [DREAM3D]((http://dream3d.bluequartz.net/?page_id=32)) is a famous tool to visualize EBSD microstructure in IPF map.  You can either use UI and create pipeline in Dream3D software but we have written a script to convert from ```npy``` format to IPF map.

You will need to need download Dream3D software from [here](http://dream3d.bluequartz.net/?page_id=32)

No need to install the software. To run Dream3D, go to ``DREAM3D-6.5.141-Linux-x86_64/bin`` and Run ``./DREAM3D``. You wont need to run GUI for this script.

To generate IPF images from Numpy files

Run 
```
./npy_to_ipf.sh
``` 

The generated IPF maps will be saved at ```experiments/saved_weights/edsr_l1_ti64/results/Test_edsr_l1_ti64/Dream3D```

Make sure that you defined proper paths for saved numpy files and reference hdf file in script and npy_to_dream3d.py and dream3d_to_rgb.py.
