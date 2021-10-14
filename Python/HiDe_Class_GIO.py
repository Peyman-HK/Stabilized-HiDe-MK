# HiDe-MK
import time
import math
import csv
import sys, os
import pickle
import random
import pandas as pd
import numpy as np
import tensorflow as tf
print('The TF version is {}.'.format(tf.__version__))
import keras
print('The Keras version is {}.'.format(keras.__version__))

from keras import metrics, models, layers
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras import regularizers, optimizers
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

#tf.compat.v1.enable_eager_execution()
indx = 1
batch_size = 1024
validation_split = 0
FILTER = 8;
Kernel_Size = 5; 
STRIDE = 5; 
################
Param_Space = pd.read_csv('/oak/stanford/groups/zihuai/Peyman/DeepPinks/Mul_DP/APOE_Proj/csv excel/Param_SKAT4.csv', header = 0);
################

################################################################################
ML_Type = sys.argv[1]  # Reg or Class
Gene_Type = sys.argv[2]  # Rare or Common or Both
Feature_Size = sys.argv[3]  # Feature size {50, 100, 200, 400, 600, 800, 1000}
print(ML_Type)
print(Gene_Type)
#print("Size of the Columns/Features is: %d." %(Feature_Size))
dataDir2 = '/oak/stanford/groups/zihuai/Peyman/DeepPinks/Mul_DP/CODE_Main/'
dataDir1 = '/oak/stanford/groups/zihuai/Peyman/DeepPinks/Mul_DP/'
dataDir = dataDir1 + str(ML_Type) + '/' + str(Gene_Type) + '/' + str(Feature_Size) + '/'
outputDir = dataDir;
print(dataDir)
################################################################################

prt1 = 'X_orig_'+str(indx)+'.csv'
prt2 = 'X_ko1_'+str(indx)+'.csv'; prt3 = 'X_ko2_'+str(indx)+'.csv'; prt4 = 'X_ko3_'+str(indx)+'.csv'
prt5 = 'X_ko4_'+str(indx)+'.csv'; prt6 = 'X_ko5_'+str(indx)+'.csv';
prt7 = 'Y_'+str(indx)+'.csv'; prt8 = 'Beta_'+str(indx)+'.csv';
output_path1 = dataDir + prt1
output_path2 = dataDir + prt2; output_path3 = dataDir + prt3; output_path4 = dataDir + prt4;
output_path5 = dataDir + prt5; output_path6 = dataDir + prt6;
output_path7 = dataDir + prt7; output_path8 = dataDir + prt8

X_orig = pd.read_csv(output_path1, header = None).values.astype(np.float64);
X_ko1 = pd.read_csv(output_path2, header = None).values.astype(np.float64);
X_ko2 = pd.read_csv(output_path3, header = None).values.astype(np.float64);
X_ko3 = pd.read_csv(output_path4, header = None).values.astype(np.float64);
X_ko4 = pd.read_csv(output_path5, header = None).values.astype(np.float64);
X_ko5 = pd.read_csv(output_path6, header = None).values.astype(np.float64);
Y = pd.read_csv(output_path7, header = None).values.astype(np.float64);
Beta = pd.read_csv(output_path8, header = None).values.astype(np.float64);

print("Size of the original feature is: %d x %d." %(X_orig.shape))
print("Size of the knockoff feature is: %d x %d." %(X_ko1.shape))
print("Size of the target is: %d x %d." %(Y.shape))
print("Size of the output weight is: %d x %d." %(Beta.shape))

X_all = np.concatenate((X_orig,X_ko1,X_ko2,X_ko3,X_ko4,X_ko5),axis=1);

Num_knock = 5;
bias = True;

num_row = X_all.shape[0];
X_dim = X_all.shape[1];

x3D_all = np.zeros((num_row, X_orig.shape[1] , Num_knock+1));
x3D_all[:, :, 0] = X_orig;
x3D_all[:, :, 1] = X_ko1; 
x3D_all[:, :, 2] = X_ko2;
x3D_all[:, :, 3] = X_ko3; 
x3D_all[:, :, 4] = X_ko4; 
x3D_all[:, :, 5] = X_ko5; 
x3D_all.shape

# Shuffle dataset
#np.random.shuffle(x3D_all)
#del X_ko1,X_ko2,X_ko3,X_ko4,X_ko5
x3D_all = (x3D_all-np.mean(x3D_all))/np.std(x3D_all);

############ Zero-Padding ###############################
print(f'The kernel size or number of neurons per group is: {Kernel_Size}')
Var_dim = x3D_all[:, :, 0].shape[1]
print(f'The total number of dimensions is: {Var_dim}')
res = Var_dim%Kernel_Size
print(f'The residual is: {res}')

Num_group = int(x3D_all.shape[1]/STRIDE); 
Size_group1 = Num_group*Kernel_Size
print(f'The total size of neurons for first locally connected layer is: {Size_group1}')

print(f'number of features/columns after zero padding: {x3D_all.shape[1]}')
#########################################################
pVal = x3D_all.shape[1];
Num_instance = x3D_all.shape[0];
seed = 457
bias = True;


def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))

def getModelLocalEq(pVal, coeff1, lr):
    print('The current coeff is')
    print(coeff1)
    
    input = Input(name='Input1', shape=(pVal, Num_knock+1));
    show_layer_info('Input1', input);

    Dropout1 = Dropout(0.6)(input)
    show_layer_info('Dropout', Dropout1);

    local1 = LocallyConnected1D(1,1, use_bias=bias, kernel_initializer=Constant(value=0.1), activation='elu', kernel_regularizer=tf.keras.regularizers.l1(coeff1))(Dropout1); 
    show_layer_info('LocallyConnected1D', local1); 

    Dropout2 = Dropout(0.6)(local1)
    show_layer_info('Dropout', Dropout2);  
       
    local2 = LocallyConnected1D(8,Kernel_Size, strides=STRIDE, use_bias=bias,kernel_initializer='glorot_normal', activation='elu')(Dropout2);
    show_layer_info('LocallyConnected1D', local2); 

    #max_pool_1d_2 = tf.keras.layers.AveragePooling1D(pool_size=Kernel_Size,strides=STRIDE, padding='same')(local2)
    #show_layer_info('MaxPooling1D', max_pool_1d_2); 

    flat = Flatten()(local2); 
    show_layer_info('Flatten', flat);            

    dense1 = Dense(30, activation='elu',use_bias=bias,kernel_initializer='glorot_normal')(flat); #, kernel_regularizer=tf.keras.regularizers.l1(coeff1)
    show_layer_info('Dense', dense1);  
   
    out_ = Dense(1, activation='sigmoid', use_bias=bias, kernel_initializer='glorot_normal')(dense1) 
    show_layer_info('Dense', out_)

    opt = Adam(lr=lr) 
    model = Model(inputs=input, outputs=out_)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.AUC()]) 

    return model


def train_DNN(model, X, y, callbacks):
    num_sequences = len(y);
    num_positives = np.sum(y);
    num_negatives = num_sequences - num_positives;

    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0,validation_split=validation_split, class_weight={True: num_sequences / num_positives, False: num_sequences / num_negatives}, callbacks=get_callbacks());
    return model;

def test_DNN(model, X, y):
    return roc_auc_score(y, model.predict(X));

def predict_DNN(model, X, y):
    return model.predict(X);


num_folds = 5       
kf = KFold(n_splits=num_folds)
all_kfold = []
kf.get_n_splits(x3D_all)
print(kf)  
KFold(n_splits=num_folds, random_state=869, shuffle=True)
pred_train = []
pred_test = []


num_coef = 5 #len(CoeffS)

 
SS = (num_folds, num_coef)
AUC_te = np.zeros(SS)
# Define per-fold score container
MSE_per_fold = []

DNN = getModelLocalEq(x3D_all.shape[1], 1e-8, 1e-9);   # lrS[int(val)]
DNN.summary()

################################################################################

#####################     Re-Train with best parameters    #####################

################################################################################  

from keras.utils.layer_utils import count_params
trainable_count = count_params(DNN.trainable_weights)
from keras.layers import Dense, Input, Concatenate, Lambda

def get_callbacks():
    #callbacks =[EarlyStopping(monitor='val_loss', patience = 90, verbose=2, mode='min')]
    callbacks =[EarlyStopping(monitor='loss', patience = 90, verbose=0, mode='min')]
    return callbacks


start = time.time()

for ii in range(0, num_coef): #val in lrS:
    lr = Param_Space['Learning_Rate'].iat[ii]
    CoeffS = Param_Space['L1_Norm'].iat[ii]
    num_epochs = int(Param_Space['Epoch'].iat[ii])
    #print('The value of Coefficients')
    #print(CoeffS[int(ii)])
    print("_" * 20) 
    print("lr, CoeffS, num_epochs, ii")
    print([lr, CoeffS, num_epochs, ii])
    print("_" * 20) 
    
    fold_no = 1
    seed = 24 * (2*indx+1)
    print(seed)
    for train_index, test_index in kf.split(x3D_all):

        X_train, X_test = x3D_all[train_index], x3D_all[test_index]
        y_train, y_test = Y[train_index], Y[test_index] 
        
        ## learning
        DNN = getModelLocalEq(pVal, CoeffS, lr);   # lrS[int(val)]
        print("_" * 50) 
        print(f'Training for fold {fold_no} ...') 
        
        print("_" * 20) 
        print("lr, CoeffS, num_epochs, ii")
        print([lr, CoeffS, num_epochs, ii])
        print("_" * 20) 
        
        #myCallback = My_Callback(outputDir, pVal);
        DNN = train_DNN(DNN, X_train, y_train, callbacks=get_callbacks); #, myCallback
       
        y_score_te = predict_DNN(DNN, X_test, y_test); 
        #print('True test values and Predicted test values')
        X_Test_True_Pred = np.concatenate((y_test,y_score_te),axis=1)                     
        #print(X_Test_True_Pred)
        AUC_te[fold_no-1, ii] = roc_auc_score(y_test, y_score_te)        
        
        print(f'AUC Test Score for fold {fold_no} and L1 Regularized {CoeffS}: {AUC_te[fold_no-1, ii]}')
        #MSE_per_fold.append(AUC_te[fold_no-1, ii])    
        del DNN
        fold_no = fold_no + 1         

end = time.time()
print(end - start)

CPU_Time = end - start

print(AUC_te)
#Average_AUC = np.mean(AUC_te,axis=0)
Average_AUC = AUC_te.mean(axis=0) #column means
print(Average_AUC)
#Best_coeff = CoeffS[np.argmax(Average_AUC)]
Opt_ind = np.argmax(Average_AUC)
print(Opt_ind)

Best_coeff = Param_Space['L1_Norm'].iat[Opt_ind]
Opt_epochs = int(Param_Space['Epoch'].iat[Opt_ind])
Best_LR = Param_Space['Learning_Rate'].iat[Opt_ind]

print("_" * 20) 
print("Best_LR, Best_coeff, Opt_epochs, Opt_ind")
print([Best_LR, Best_coeff, Opt_epochs, Opt_ind])
print("_" * 20) 

print('Retrain with best parameters')

Gradients_All=np.zeros((num_row,pVal,Num_knock+1));

#tf.enable_eager_execution()
counter=Opt_epochs;
class My_Callback(keras.callbacks.Callback):
    def __init__(self, outputDir, pVal):
        self.outputDir = outputDir;
        self.pVal = pVal;
        
        print(self.outputDir);      
    def on_epoch_end(self, epoch, logs={}):
        global counter
        print(epoch+1)
        #if epoch+1 == Opt_epochs:
        counter = counter-1;
        print("counter is",counter)   

    def on_batch_end(self, batch, logs={}): 
        global counter
        if counter == 1:
            #print("Yes") 
            print(batch)
            x_tensor = tf.convert_to_tensor(x3D_all[(batch*batch_size):(batch+1)*batch_size,:,:], dtype=tf.float32)
            with tf.GradientTape() as t:
                t.watch(x_tensor)
                output = DNN(x_tensor)
                print("size of output is:",output.shape)
                #print(output)
            gradients = t.gradient(output, x_tensor)
            print("size of gradient is:",gradients.shape)
            #Gradients_All.append(gradients) 
            Gradients_All[(batch*batch_size):(batch+1)*batch_size,:,:]=gradients;
    #return gradients

validation_split = 0

def train_DNN(model, X, y, myCallback):
    num_sequences = len(y);
    num_positives = np.sum(y);
    num_negatives = num_sequences - num_positives;

    model.fit(X, y, epochs=Opt_epochs, batch_size=batch_size, verbose=0, class_weight={True: num_sequences / num_positives, False: num_sequences / num_negatives}, callbacks=[myCallback]);
    return model;

DNN = getModelLocalEq(pVal, Best_coeff, Best_LR);   # lrS[int(val)]
myCallback = My_Callback(outputDir, pVal);
DNN_trained = train_DNN(DNN, x3D_all, Y, myCallback);

y_score = predict_DNN(DNN_trained, x3D_all, Y); 
AUC_All = roc_auc_score(Y, y_score)       
print(AUC_All)

Grad2 = Gradients_All;
#Grad2 = Gradients_All[0:9800,:,:]

avg_all = np.zeros((pVal,Num_knock+1));
for row_ind in range(Grad2.shape[1]):
    for col_ind in range(Grad2.shape[2]):
        avg_all[row_ind,col_ind] = np.mean(Grad2[:,row_ind,col_ind]) 

Feat_Import = np.hstack((avg_all[:,0],avg_all[:,1],avg_all[:,2],avg_all[:,3],avg_all[:,4],avg_all[:,5]));

#Feat_Import_abs = np.abs(Feat_Import);

np.savetxt('FI_Class' + str(indx)+ '.csv', Feat_Import, delimiter=",")

print("Opt_epochs, Best_coeff, Best_LR, AUC_All, Average_AUC, np.max(Average_AUC), trainable_count,CPU_Time, indx")
METRICS_HMDP_New = [Opt_epochs, Best_coeff, Best_LR, AUC_All, Average_AUC, np.max(Average_AUC), trainable_count,CPU_Time, indx]; 

filename_metric = 'METRICS_HMDP_CLS_B' + str(Feature_Size) + '.csv'

with open(os.path.join(dataDir2, filename_metric), "a+") as fp:
    wr = csv.writer(fp, dialect='excel'); 
    wr.writerow(METRICS_HMDP_New);
try:
    os.stat(outputDir)
except:
    os.mkdir(outputDir)

print(METRICS_HMDP_New)
print('Done!')     


