# MVSPIN
This repo is modified from [SPIN](https://github.com/nkolot/SPIN). The required dependencies have been shown in SPIN. If you want to run the code, please install all the dependencies correctly.
## Fetch data
SPIN also gives the necessary basic models including the GMM prior, male and female SMPL models. Please refer to [SPIN](https://github.com/nkolot/SPIN).
## Process dataset
We use two public dataset in our experiments: [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/).
Before you start the experiment, use ```dataset/preporecess/h36m_train.py``` and ```dataset/preporecess/mpi_inf_3dhp_mv.py``` to get training dataset for H36M and MPI-INF-3DHP. 

The testing dataset can be obtained by ```dataset/preporecess/h36m.py``` and ```dataset/preporecess/mpi_inf_3dhp_mv.py```.

After preprocessing, ```.npz``` files containing the names of images and joints will be obtained. 
## Training
After generating the training dataset, we can run the training code. The options for training can be found at ```utils/train_options```. The initialization of the CNN is the trained model from SPIN. You can also download our trained model at [here](https://drive.google.com/drive/folders/1kvpEyzXz8k5vhmLQnlLzQD7Qf-xvjY_T?usp=sharing). The example for training is:
```
python3 train.py --name train_example --pretrained_checkpoint=data/model_checkpoint.pt --run_smplify
```
## Testing
We will obtian trained models at ```logs/train_example/checkpoints/```. Then, we can run the testing codel to evaluate the results. For Human3.6M, the testing code is:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
If you want to test MPI-INF-3DHP, you need to set ```--dataset``` as ```mpi-inf-3dhp```. 
## Visulization
If you save the testing results, you can use to visuliza the results of testing data by using the code:

But you need to install [OpenDr](https://github.com/mattloper/opendr/wiki) to visulize the results.
