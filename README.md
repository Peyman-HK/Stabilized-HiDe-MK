# "De-randomized hierarchical deep neural networks with multiple knockoffs (De-randomized HiDe-MK)"

## Overview 
Deep neural networks (DNN) have been used successfully in many scientific problems for their high prediction accuracy, but their application to genetic studies remains challenging due to their poor interpretability. We consider the problem of scalable, robust variable selection in DNN for the identification of putative causal genetic variants in genome sequencing studies. We identified a pronounced randomness in feature selection in DNN due to its stochastic nature, which may hinder interpretability and give rise to misleading results. We propose an interpretable neural network model, stabilized using ensembling, with controlled variable selection for genetic studies. The merit of the proposed method includes: (1) flexible modelling of the non-linear effect of genetic variants to improve statistical power; (2) multiple knockoffs in the input layer to rigorously control false discovery rate; (3) hierarchical layers to substantially reduce the number of weight parameters and activations to improve computational efficiency; (4) de-randomized feature selection to stabilize identified signals. We evaluated the proposed method in extensive simulation studies and applied it to the analysis of Alzheimer disease genetics. We showed that the proposed method, when compared to conventional linear and nonlinear methods, can lead to substantially more discoveries.

## Repo Contents


## System Requirements
### Hardware requirements

### Software requirements







This repository contains code samples for "Deep neural networks with controlled variable selection for the identification of putative causal genetic variants". The method is a novel hierarchical deep neural network which is equipped with multiple sets of knockoffs with FDR guarantees. 

![main flow](/../main/Images/Flowchart.jpg?raw=true "HiDe-MK pipeline")

The code is written for Python 3.6, Keras 2.4 and Tensorflow 2.1. It may need modification for compatibility with earlier or later versions of the libraries. We also examined the code for Tensorflow 2.6 and Keras 2.6 and it worked well with no error. 

To obtain the FDR and power through the feature importance score learned by HiDe-MK, you should run the R script FDR_Pow_MK.R. The target FDR levels are set from 0.01 to 0.20 with step size 0.01. 

The simulation datasets were also written in R programming language for both classification and regression problems, reproducible by Gen_CLS_LRS.R and Gen_Reg_LRS.R. 

The KnockoffScreen method was first proposed in Nature communications journal at: https://www.nature.com/articles/s41467-021-22889-4

Any question, please feel free to ask me at: peymanhk@stanford.edu
