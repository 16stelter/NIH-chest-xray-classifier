# NIH-chest-xray-classifier

This repository contains multiple different CNN implementations to tackle the classification problem for the NIH  chest xray dataset. 
The datset can be found [here](https://www.kaggle.com/nickuzmenkov/nih-chest-xrays-tfrecords). 
The repository also contains visualization and dataset manipulation functionalities, as well as the results of some of the experiments run with the networks.

## Requirements

You can find the required python libraries in the ```requirements.txt``` file.

## What is where?

```tf_cnn.py``` - A standard tensorflow CNN with 6 convolutional and 2 pooling layers.

```custom_cnn.py``` - A CNN implemented from scratch, following the same structure as the network above.

```mobilenet.py``` - A CNN that uses [MobileNet](https://arxiv.org/abs/1704.04861) as a base.

```optuna_optimization.py``` - Code that creates an [optuna](https://optuna.org/) study and optimizes a CNN.

```optimized_model.py``` - The network that returned the best results in the optuna study.


## How to run

All networks are set up to first train, then predict and calculate a classification report on execution. 
Simply run the any file by using ```python3 filename.py```.
The networks require the dataset mentioned above to run. 
By default, the dataset is expected in a folder called ```data```, located in the same folder where the python script is run.
The optuna optimizer expects a calibration dataset in a folder called ```cal_ds```. 
This dataset can be generated using the ```create_balanced_dataset``` method of the ```utility.py``` file.
The full sized dataset could also be substituted in, but training will take significantly longer.

Parameters can currently only be manipulating by directly changing the source code.
All relevant parameters can be found in the init function of each class.
The visualizer may need to be modified, as some necessary files for the plots are not present.
