# "De-randomized hierarchical deep neural networks with multiple knockoffs (De-randomized HiDe-MK)"

## Overview 
Deep neural networks (DNN) have been used successfully in many scientific problems for their high prediction accuracy, but their application to genetic studies remains challenging due to their poor interpretability. We consider the problem of scalable, robust variable selection in DNN for the identification of putative causal genetic variants in genome sequencing studies. We identified a pronounced randomness in feature selection in DNN due to its stochastic nature, which may hinder interpretability and give rise to misleading results. We propose an interpretable neural network model, stabilized using ensembling, with controlled variable selection for genetic studies. The method is a novel hierarchical deep neural network which is equipped with multiple sets of knockoffs with FDR guarantees. The main pipelines of the method is displayed in the following:

![main flow](/../main/Images/Flowchart.jpg?raw=true "HiDe-MK pipeline")


## Repo Contents

R: R simulation codes

Python: python codes

Manuals: Package manuals for help in both R and python


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
install.packages(c('glmnet', 'data.matrix', 'knockoff'))

## Results

In the folder manuals, we explained step by step how to run the code to observe results from both learning through stabilized HiDe-MK python code, and obtaining FDR-Power through R scripts. To obtain the FDR and power through the feature importance score learned by stabilized HiDe-MK, you should run the R script FDR_Pow_MK.R. The target FDR levels are set from 0.01 to 0.20 with step size 0.01. 


We also uploaded a tabel, "Table_Summary_TopMed_Imputed_Median.csv", obtained through the real data analysis which includes feature importance scores for each genetic variant. Through the "Manhattan plot.ipyng" and using jupyter notebook, the following plot can be displayed: 

![Main flow](/../main/Images/Manhattan_plot.jpg?raw=true "Manhattan plot")


## Citations

The paper titled by "Deep neural networks with controlled variable selection for the identification of putative causal genetic variants" is in Arxiv:

https://arxiv.org/abs/2109.14719

To extract the knockoff features we used KnockoffScreen method which was first proposed in Nature communications journal at: https://www.nature.com/articles/s41467-021-22889-4

Please feel free to write me your questions at: peymanhk@stanford.edu


