# Stabilized HiDe-MK
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
from tensorflow.keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping
from keras.initializers import Constant

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

################
Param_Space = pd.read_csv('C:/Users/15043/Stabilized_HiDe_MK/Param_SKAT2.csv', header = 0);
################


num_epochs = int(100)
num_param = int(20) #len(CoeffS)
cuttoff_point = int(50)#num_epochs*num_param


def _index_predictions(predictions, labels):
    '''
    Indexes predictions, a [batch_size, num_classes]-shaped tensor,
    by labels, a [batch_size]-shaped tensor that indicates which
    class each sample should be indexed by.

    Args:
        predictions: A [batch_size, num_classes]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The tensor used to index predictions, in one-hot encoding form.
    Returns:
        A tensor of shape [batch_size] representing the predictions indexed by the labels.
    '''
    current_batch_size = tf.shape(predictions)[0]
    sample_indices = tf.range(current_batch_size)
    sparse_labels  = tf.argmax(labels, axis=-1)
    indices_tensor = tf.stack([sample_indices, tf.cast(sparse_labels, tf.int32)], axis=1)
    predictions_indexed = tf.gather_nd(predictions, indices_tensor)
    return predictions_indexed


indx = 1
batch_size = 1024
validation_split = 0
FILTER = 8;
KERNEL = 25; 
STRIDE = 25; 


#######################################
outputDir = 'C:/Users/15043/Stabilized_HiDe_MK/'
dataDir = 'C:/Users/15043/Stabilized_HiDe_MK/Dicho/'
#######################################
        
prt0 = 'X_orig_'+str(indx)+'.csv'
prtk1 = 'X_ko1_'+str(indx)+'.csv'; prtk2 = 'X_ko2_'+str(indx)+'.csv'; prtk3 = 'X_ko3_'+str(indx)+'.csv'
prtk4 = 'X_ko4_'+str(indx)+'.csv'; prtk5 = 'X_ko5_'+str(indx)+'.csv';
prtY = 'Y_'+str(indx)+'.csv'; prtBeta = 'Beta_'+str(indx)+'.csv';
output_pathX0 = dataDir + prt0
output_pathK1 = dataDir + prtk1; output_pathK2 = dataDir + prtk2; output_pathK3 = dataDir + prtk3;
output_pathK4 = dataDir + prtk4; output_pathK5 = dataDir + prtk5;
output_pathY = dataDir + prtY; output_path_Beta = dataDir + prtBeta

X_orig = pd.read_csv(output_pathX0, header = None).values.astype(np.float64);
X_ko1 = pd.read_csv(output_pathK1, header = None).values.astype(np.float64);
X_ko2 = pd.read_csv(output_pathK2, header = None).values.astype(np.float64);
X_ko3 = pd.read_csv(output_pathK3, header = None).values.astype(np.float64);
X_ko4 = pd.read_csv(output_pathK4, header = None).values.astype(np.float64);
X_ko5 = pd.read_csv(output_pathK5, header = None).values.astype(np.float64);
Y = pd.read_csv(output_pathY, header = None).values.astype(np.float64);
Beta = pd.read_csv(output_path_Beta, header = None).values.astype(np.float64);


print("Size of the original feature is: %d x %d." %(X_orig.shape))
print("Size of the knockoff feature is: %d x %d." %(X_ko1.shape))
print("Size of the target is: %d x %d." %(Y.shape))
print("Size of the output weight is: %d x %d." %(Beta.shape))

Num_knock = 5;
bias = True;
num_row = X_orig.shape[0];
X_dim = X_orig.shape[1]*(Num_knock+1);

x3D_all = np.zeros((num_row, X_orig.shape[1] , Num_knock+1));
x3D_all[:, :, 0] = X_orig;
x3D_all[:, :, 1] = X_ko1; 
x3D_all[:, :, 2] = X_ko2;
x3D_all[:, :, 3] = X_ko3; 
x3D_all[:, :, 4] = X_ko4; 
x3D_all[:, :, 5] = X_ko5; 
x3D_all.shape

######## Zero-Padding #########
Orig_dim = X_orig.shape[1]
res = Orig_dim%STRIDE
print(f'The residual for zero-padding is: {res}')
if res != 0:
    new_row = x3D_all.shape[0];
    new_col = STRIDE -res;
    x_new = np.zeros((new_row, new_col,Num_knock+1));
    x3D_all = np.hstack((x3D_all,x_new)); 
    
print(f'number of features/columns after zero padding: {x3D_all.shape[1]}')
###############################

del X_orig, X_ko1, X_ko2, X_ko3, X_ko4, X_ko5

pVal = x3D_all.shape[1];
Num_instance = x3D_all.shape[0];
seed = 457

def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))

def getModelLocalEq(pVal, coeff1, lr):
    print('The current coeff is')
    print(coeff1)
    
    input = Input(name='Input1', shape=(pVal, Num_knock+1));
    show_layer_info('Input1', input);

    local1 = LocallyConnected1D(1,1, use_bias=bias, kernel_initializer=Constant(value=0.1), activation='elu', kernel_regularizer=tf.keras.regularizers.l1(coeff1))(input); 
    show_layer_info('LocallyConnected1D', local1); 

    Dropout1 = Dropout(0.1)(local1)
    show_layer_info('Dropout', Dropout1);  
       
    local2 = LocallyConnected1D(FILTER, KERNEL, strides=STRIDE, use_bias=bias,kernel_initializer='glorot_normal', activation='elu')(Dropout1);
    show_layer_info('LocallyConnected1D', local2); 

    flat = Flatten()(local2); 
    show_layer_info('Flatten', flat);            

    dense1 = Dense(50, activation='elu',use_bias=bias,kernel_initializer='glorot_normal')(flat); 
    show_layer_info('Dense', dense1);  
   
    out_ = Dense(1, activation='linear', use_bias=bias, kernel_initializer='glorot_normal')(dense1) 
    show_layer_info('Dense', out_)

    opt = Adam(learning_rate=lr) 
    model = Model(inputs=input, outputs=out_)
    
    model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.AUC()]) 

    return model


def test_DNN(model, X, y):
    return mean_squared_error(y, model.predict(X));

def predict_DNN(model, X, y):
    return model.predict(X);


num_folds = 5       
kf = KFold(n_splits=num_folds, random_state=869, shuffle=True)
kf.get_n_splits(x3D_all)
print(kf)  
#KFold(n_splits=num_folds, random_state=869, shuffle=True)

train_loss_All=[]; train_AUC_All=[];Val_loss_All=[];Val_mse_All=[];
AA = (num_param, num_epochs)
#BB = (num_param)
SS = (num_param,num_folds)
AUC_te = []; 
AVG_Val_Loss = np.zeros(AA)
AVG_Val_mse = np.zeros(AA)
#AVG_Val_mse = np.zeros(num_param)
print('The shape of AVG_Val_Loss is : ',AVG_Val_Loss.shape)
print('The shape of AVG_Val_mse is : ',AVG_Val_mse.shape)
print('The shape of SS is : ',SS)
print('The shape of AA is : ',AA)


DNN = getModelLocalEq(x3D_all.shape[1], 1e-3, 1e-3);   
DNN.summary()


def get_callbacks():
    callbacks =[EarlyStopping(monitor='loss', patience = 1000, verbose=0, mode='min')]
    return callbacks


np.random.seed(seed)
start = time.time()

for param_counter in range(0, num_param): # Loop for different parameter set
    print('The current paramter counter is: ',param_counter+1)
    lr = Param_Space['Learning_Rate'].iat[param_counter]
    CoeffS = Param_Space['L1_Norm'].iat[param_counter]
    print("*" * 40) 
    print("lr, CoeffS, num_epochs, Num_param")
    print([lr, CoeffS, num_epochs, param_counter+1])
    print("*" * 40)     
    fold_no = 0
    Val_loss_All=[]; Val_mse_All = [];    
    for train_index, val_index in kf.split(x3D_all):  # Loop for different 5 folds            
        X_train, X_val = x3D_all[train_index], x3D_all[val_index]
        y_train, y_val = Y[train_index], Y[val_index]         
        ## learning
        print("_" * 50) 
        print(f'Training for fold {fold_no+1} ...') 
        print('The current paramter counter is: ',param_counter+1) 
        DNN = getModelLocalEq(x3D_all.shape[1], CoeffS, lr);
        history = DNN.fit(X_train, y_train, epochs=num_epochs, 
                          batch_size=batch_size, validation_data=(X_val, y_val), verbose=0);              
        print(history.history.keys())
        Val_loss = history.history[list(history.history.keys())[2]]
        Val_mse = history.history[list(history.history.keys())[3]]  
        print(f'AUC validation Score for fold {fold_no} and L1 Regularized {CoeffS} is: {Val_mse, param_counter+1}')
        Val_mse_All.append(Val_mse)
        Val_loss_All.append(Val_loss)     
        fold_no = fold_no + 1         
    print("the Shape of Val Loss All for the current 5-fold is: ",np.asarray(Val_loss_All).shape) 
    AVG_Val_Loss[param_counter] = np.mean(np.array(Val_loss_All),axis=0) 
    AVG_Val_mse[param_counter] = np.mean(np.array(Val_mse_All),axis=0) 
#############    


end = time.time()

CPU_Time = end - start
print(CPU_Time)

CoeffS = np.array(Param_Space['L1_Norm'][:num_param])

temp_coeff = np.tile(CoeffS.transpose(), (num_epochs,1)).T

Lr = np.array(Param_Space['Learning_Rate'][:num_param])

temp_Lr = np.tile(Lr.transpose(), (num_epochs,1)).T

vect_val = AVG_Val_Loss.ravel() # vectorize a numpy array

vect_coef = temp_coeff.ravel() # vectorize a numpy array

vect_lr = temp_Lr.ravel() # vectorize a numpy array

temp_sort_ind = np.argsort(vect_val) 

maximum_loss = np.max(vect_val)
minimum_loss = np.min(vect_val)
W_FIs = np.zeros((1, num_param*num_epochs))
for i in range(num_param*num_epochs):
    W_FIs[0,i]=(maximum_loss - vect_val[i])/(maximum_loss - minimum_loss)
  
W_optimal = W_FIs[0,temp_sort_ind]

truncated_sort_ind = temp_sort_ind[:cuttoff_point]

vect_opt_lr = vect_lr[temp_sort_ind][:cuttoff_point]

vect_opt_coef = vect_coef[temp_sort_ind][:cuttoff_point]

q, r = divmod((truncated_sort_ind+1), num_epochs)

temp2 = q+(r/num_epochs)

category_param = np.ceil(temp2)

epoch_trunc = np.zeros((cuttoff_point))
q, r = divmod((truncated_sort_ind+1), category_param)

res = (truncated_sort_ind+1)%num_epochs

for i in range(0,cuttoff_point):
    if res[i] != 0:
        epoch_trunc[i] = res[i]
    else:
        epoch_trunc[i] = num_epochs

For_FI_callback = np.column_stack((vect_opt_lr,vect_opt_coef,epoch_trunc,W_optimal[:cuttoff_point]))


print(For_FI_callback)


print('Retrain with best parameters')
Feat_Import = [];
class My_Callback2(keras.callbacks.Callback):
    def __init__(self, outputDir, pVal, epoch_param_current):
        self.outputDir = outputDir;
        self.pVal = pVal;
        self.epoch_param_current = epoch_param_current;
        print(self.outputDir);
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1) in epoch_param_current:  
            #print('Calc grad here for epoch: ', epoch+1)
            x3D_allT = tf.convert_to_tensor(x3D_all, dtype=tf.float32) 
            with tf.GradientTape() as tape:
                tape.watch(x3D_allT)
                predictions = DNN(x3D_allT)
                predictions_indexed = _index_predictions(predictions, Y)
            gradients = tape.gradient(predictions_indexed, x3D_allT)
            temp_g = gradients.numpy() 
            #print('Shape of temp_g is : ',temp_g.shape)
            avg_all = np.zeros((pVal,Num_knock+1));
            for row_ind in range(temp_g.shape[1]):
                for col_ind in range(temp_g.shape[2]):
                    avg_all[row_ind,col_ind] = np.mean(temp_g[:,row_ind,col_ind])         
            #print('Shape of avg_all is : ',avg_all.shape)        
            Feat_Import.append(np.hstack((avg_all[:,0],avg_all[:,1],avg_all[:,2],avg_all[:,3],avg_all[:,4],avg_all[:,5])))
################

validation_split = 0
def train_DNN(model, X, y, myCallback):
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=1,
              validation_split=0, callbacks=[myCallback]);   
    return model; 
################################################################################



for i in range(0,num_param):
    print('the current parameter to check is: ',i+1)
    lr = Param_Space['Learning_Rate'].iat[i]
    CoeffS = Param_Space['L1_Norm'].iat[i]
    temp = For_FI_callback[(For_FI_callback[:,0]==lr) & (For_FI_callback[:,1]==CoeffS)]
    sorted_filtered_param = temp[temp[:, 2].argsort()]
    epoch_param_current = sorted_filtered_param[:,2]
    print('The length of the epochs for the current paramter is',len(epoch_param_current))
    if len(epoch_param_current)==0: 
        print('The size of epoch for this current param is zero')            
    else:
        DNN = getModelLocalEq(pVal, CoeffS, lr);   # lrS[int(val)]
        myCallback = My_Callback2(outputDir, pVal, epoch_param_current);
        trained_DNN = train_DNN(DNN, x3D_all, Y, myCallback);

Temp = np.array(np.transpose(Feat_Import)).dot(np.transpose(For_FI_callback[:,3]))
#print("Size of Temp is",Temp.shape)  
FI_final = Temp/np.sum(W_FIs)
FI_final

np.savetxt(outputDir + 'FI_Reg_' + str(indx)+ '.csv', FI_final, delimiter=",")

print('Done!')  

