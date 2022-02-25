
# "Stabilized hierarchical deep neural networks with multiple knockoffs (Stabilized HiDe-MK)"

## Overview 
Deep neural networks (DNN) have been used successfully in many scientific problems for their high prediction accuracy, but their application to genetic studies remains challenging due to their poor interpretability. We consider the problem of scalable, robust variable selection in DNN for the identification of putative causal genetic variants in genome sequencing studies. We identified a pronounced randomness in feature selection in DNN due to its stochastic nature, which may hinder interpretability and give rise to misleading results. We propose an interpretable neural network model, stabilized using ensembling, with controlled variable selection for genetic studies. The method is a novel hierarchical deep neural network which is equipped with multiple sets of knockoffs with FDR guarantees. The main pipelines of the method is displayed in the following:

![main flow](/../main/Images/Flowchart.jpg?raw=true "HiDe-MK pipeline")


## Repo Contents

R: R simulation codes

Python: python codes

Manual: Package manual for help in both R and python


## System Requirements
### Hardware requirements
This package requires only a standard computer with about 4 GB of RAM. For optimal performance, we recommend a computer with the following hardware properties:

RAM: 4+ GB

CPU: 4+ cores, 3.3+ GHz

We examined the codes on both local computer and remote.

### Software requirements

#### OS Requirements

The package is tested on Windows 10 and Mac OS operating systems. We did not test it with Linux operating systems. Before setting up the HiDe-MK package, users should have R version 3.4.0 or higher, Python version 3.6.0, Keras 2.4 and Tensorflow 2.1 or Keras 2.6 and Tensorflow 2.6. Several packages and dependencies should be set up from CRAN and Python libraries as well.

## Installation Guide

### Install necessary python libraries
pip3 install tensorflow==2.6

pip3 install keras==2.6


### Install R dependencies
install.packages(c('glmnet', 'data.matrix', 'KnockoffScreen'))

## Results

In the manual' folder, we step by step explained how to run the code to obtain results for learning through stabilized HiDe-MK (python code), and results for FDR-Power (R scripts). To obtain the FDR and power through the feature importance scores learned by stabilized HiDe-MK, you should run the R script FDR_Pow_MK.R. The target FDR levels are set from 0.01 to 0.20 with step size 0.01. 


We also uploaded a tabel, "Table_summary_HF_KS_Lancet.csv", obtained through the real data analysis which includes q-values for each genetic variant. Through the "Manhattan plot.ipyng" and using jupyter notebook, the following plot is displayed: 

![Main flow](/../main/Images/Manhattan_plot.jpg?raw=true "Manhattan plot")


## Example to run the code

The following steps helps test/run the proposed HiDe-MK for a classification example (dichotomous trait):

Steps:
1. Create a folder somewhere to keep the installed libraries.
Let us call this folder: 
C:/users/De-randomized-HiDe-MK/

Download  De-randomized-HiDe-MK from Github and put its content into this folder. 

2. Install Tensorflow version 2.6
pip3 install --user tensorflow==2.6

If pip3 gives error, please try with pip or conda.  

3. Change the value for the variable 'outputDir' in line 72 of file 'HiDe_Class_GIO.py' in the path 'C:/users/hide_mk/Python' to:

outputDir = 'C:/users/De-randomized-HiDe-MK/Simulation data/class/'

This fix the address of data files.


4. Run the example python script 'HiDe_Class_GIO.py' which loads the following libraries:
tensorflow/2.6
keras/2.6
numpy
pandas
SKlearn

5- The output of this file is a csv file with name "FI_Class_1.csv" which is the vector of the feature importance scores (FIs). 

You are done with learning from data and extracting FIs. This approaximately takes an hour to two hours. 


---> How to get FDR and Power through the R scrip:  <---

6- First, install the following packages throgu the R: 

install.packages(c('glmnet', 'data.matrix', 'knockoff'))

7- Then, open the file "FDR_Pow_MK.R" throug the folder 'R' of the downloaded package. 

8- Again, change the line 'path_data' (line 22) in this file to read the FIs (which was learned by the neuo-net), Beta, and Y. 

9- Update or change the path for source('/home/users/peymanhk/PROJECT_2/Simulated/GOI/KnockoffScreen_updated.R') to where KnockoffScreen_updated.R is located in your PC of folder "De-randomized-HiDe-MK"

10- Run this R script and you get FDR and Power for different target FDR levels from 0.01 to 0.20 by step size 0.01. 

11- A csv file "Stat_stabilize_HiDe_MK_Class.csv" in the folder "Simulation data" is generated which includes a row with 40 values. 
The first 20 values are observed/empricial FDRs and the second 20 values are the emprical power for the corresponding FDRs. 

12- Done!

## Citations

The paper titled by "Deep neural networks with controlled variable selection for the identification of putative causal genetic variants" is uploaded to Arxiv:

https://arxiv.org/abs/2109.14719

To extract the knockoff features, we used KnockoffScreen method which was first proposed in journal of nature communications at: https://www.nature.com/articles/s41467-021-22889-4

Please feel free to write me your questions at: peymanhk@stanford.edu


