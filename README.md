# Code sample for "De-randomized hierarchical deep neural networks with multiple knockoffs (De-randomized HiDe-MK)"

This repository contains code samples for "Deep neural networks with controlled variable selection for the identification of putative causal genetic variants". The method is a novel hierarchical deep neural network which is equipped with multiple sets of knockoffs with FDR guarantees. 

![main flow](/../main/Images/Flowchart.jpg?raw=true "HiDe-MK pipeline")

The code is written for Python 3.6, Keras 2.4 and Tensorflow 2.1. It may need modification for compatibility with earlier or later versions of the libraries. We also examined the code for Tensorflow 2.6 and Keras 2.6 and it worked well with no error. 

To obtain the FDR and power through the feature importance score learned by HiDe-MK, you should run the R script FDR_Pow_MK.R. The target FDR levels are set from 0.01 to 0.20 with step size 0.01. 

The simulation datasets were also written in R programming language for both classification and regression problems, reproducible by Gen_CLS_LRS.R and Gen_Reg_LRS.R. 

The KnockoffScreen method was first proposed in Nature communications journal at: https://www.nature.com/articles/s41467-021-22889-4

Any question, please feel free to ask at: peymanhk@stanford.edu
