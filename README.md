# MVSPIN
This repo is modified from [SPIN](https://github.com/nkolot/SPIN). The required dependencies have been shown in SPIN. If you want to run the code, please install all the dependencies correctly.
## Fetch data
SPIN also gives the necessary basic models including the GMM prior, male and female SMPL models. Please refer to [SPIN](https://github.com/nkolot/SPIN).
## Process dataset
We use two public dataset in our experiments: [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).
Before you start the experiment, use ```dataset/preporecess/h36m_train.py``` and ```dataset/preporecess/mpi_inf_3dhp_mv.py``` to get training dataset for H36M and MPI-INF-3DHP. 

The testing dataset can be obtained by ```dataset/preporecess/h36m.py``` and ```dataset/preporecess/mpi_inf_3dhp_mv.py```.

After preprocessing, ```.npz``` files containing images and joints will be obtained. 
