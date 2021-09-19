# De-randomized Hierarchical deep neural network with multiple knockoffs (De-randomized HiDe-MK)

This repository code was written in Keras is for a novel hierarchical deep neural network which is equipped with multiple sets of knockoffs with FDR guarantees. To reproduce the results please have Keras version 2.4 and Tensorflow 2.1.  

To obtain the FDR and power through the feature importance score learned by HiDe-MK, you should run the R script. The target FDR level is from 0.10 to 0.20 with step size 0.01. THe file name is FDR_Pow_MK.R. 

The simulation datasets was also written in R programming language for both classification and regression problems with filenames gen_cls.R and gen_reg.R. 




