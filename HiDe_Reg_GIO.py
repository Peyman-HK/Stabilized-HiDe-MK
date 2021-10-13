# DeepPink
import time
import math
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
print(keras.__version__)
import random
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from keras.objectives import mae
from keras import regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from math import sqrt
from sklearn.model_selection import KFold
from keras import metrics
from sklearn import preprocessing
import sys, os
import pickle
from keras.optimizers import Adam, RMSprop,SGD
from keras import optimizers, metrics, models, layers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from scipy import stats
import sklearn
from skopt.space import Real, Integer
from keras.layers import Dense, Input, Concatenate, Lambda

print('The scikit-learn version is {}.'.format(sklearn.__version__))
#tf.compat.v1.enable_eager_execution()
print('The TF version is {}.'.format(tf.__version__))

indx = 1
batch_size = 1024
validation_split = 0
################ Load a CSV file of the tuning hyperparameters with column headers 'Learning_Rate', 'L1_Norm', and 'Epoch'
Param_Space = pd.read_csv('../Param_SKAT.csv', header = 0);
################
FILTER = 8;
Kernel_Size = 5; 
STRIDE = 5; 
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
validation_split = 0
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
Y = (Y-np.mean(Y))/np.std(Y);
############ Zero-Padding ###############################
Var_dim = X_orig.shape[1]
res = Var_dim%STRIDE
print(f'The residual for zero-padding is: {res}')
if res != 0:
    new_row = x3D_all.shape[0];
    new_col = STRIDE -res;
    x_new = np.zeros((new_row, new_col,6));
    x3D_all = np.hstack((x3D_all,x_new)); 
    
Num_group = int(x3D_all.shape[1]/STRIDE); 
print(f'Number of groups is: {Num_group}')
print(f'number of features/columns after zero padding: {x3D_all.shape[1]}')
#########################################################
pVal = x3D_all.shape[1];
Num_instance = x3D_all.shape[0];
seed = 457
#np.random.seed(seed)
bias = True;
#CoeffS = 0.001 * np.sqrt((2.0 * np.log(pVal)) / Num_instance);
#CoeffS = np.array([0.0035]);
#CoeffS = np.array([ 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3])#, 1e-6, 3e-6, 5e-6, 7e-6, 1e-5, 3e-5, 5e-5, 7e-5,

def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))

def getModelLocalEq(pVal, coeff1, lr, Drop_Rate):
    print('The current coeff is')
    print(coeff1)
    
    input = Input(name='input', shape=(pVal, Num_knock+1));
    show_layer_info('Input', input);

    Dropout1 = Dropout(Drop_Rate)(input)
    show_layer_info('Dropout', Dropout1);

    local1 = LocallyConnected1D(1,1, use_bias=bias, kernel_initializer=Constant(value=0.1), activation='elu', kernel_regularizer=tf.keras.regularizers.l1(coeff1))(Dropout1); 
    show_layer_info('LocallyConnected1D', local1); 

    Dropout2 = Dropout(Drop_Rate)(local1)
    show_layer_info('Dropout', Dropout2);    
           
    local2 = LocallyConnected1D(8,Kernel_Size, strides=STRIDE, use_bias=bias,kernel_initializer='glorot_normal', activation='elu')(Dropout2);
    show_layer_info('LocallyConnected1D', local2); 
         
    flat = Flatten()(local2); 
    show_layer_info('Flatten', flat);            

    dense1 = Dense(30, activation='elu',use_bias=bias,kernel_initializer='glorot_normal')(flat);
    show_layer_info('Dense', dense1);  
   
    out_ = Dense(1, activation='linear', use_bias=bias, kernel_initializer='glorot_normal')(dense1) 
    show_layer_info('Dense', out_)

    opt = Adam(lr=lr) 
    model = Model(inputs=input, outputs=out_)
    
    model.compile(loss='mse', optimizer=opt, metrics=['mae','mse']) 

    return model


def train_DNN(model, X, y, x_test, y_test, callbacks):
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=0, validation_data=(x_test, y_test), callbacks=get_callbacks());   
    return model;

def test_DNN(model, X, y):
    return mean_squared_error(y, model.predict(X));

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

#################################
num_coef = 12 #len(CoeffS)
#################################

SS = (num_folds, num_coef)
MSE_te = np.zeros(SS)
# Define per-fold score container
MSE_per_fold = []

DNN = getModelLocalEq(x3D_all.shape[1], 1e-8, 1e-9, 0.2);  # lrS[int(val)]
DNN.summary()

################################################################################

#####################     Re-Train with best parameters    #####################

################################################################################  

from keras.utils.layer_utils import count_params
trainable_count = count_params(DNN.trainable_weights)
from keras.layers import Dense, Input, Concatenate, Lambda

def get_callbacks():
    callbacks =[EarlyStopping(monitor='val_loss', patience = 200, verbose=0, mode='min')]
    return callbacks

start = time.time()
for ii in range(0, num_coef): #val in lrS:
    print(ii)
    lr = Param_Space['Learning_Rate'].iat[ii]
    CoeffS = Param_Space['L1_Norm'].iat[ii]
    num_epochs = int(Param_Space['Epoch'].iat[ii])
    Drop_Rate = Param_Space['Drop_Rate'].iat[ii]+0.2 

    #print('The value of Coefficients')
    #print(CoeffS[int(ii)])
    print("_" * 20) 
    print("lr, CoeffS, num_epochs, Drop_Rate, ii")
    print([lr, CoeffS, num_epochs, Drop_Rate, ii])
    print("_" * 20) 
   
    fold_no = 1
    seed = 457 #24 * (2*indx+1)
    print(seed)
    #np.random.seed(seed)
    for train_index, test_index in kf.split(x3D_all):

        X_train, X_test = x3D_all[train_index], x3D_all[test_index]
        y_train, y_test = Y[train_index], Y[test_index] 
        
        ## learning
        DNN = getModelLocalEq(pVal, CoeffS, lr, Drop_Rate);   # lrS[int(val)]
        print("_" * 50) 
        print(f'Training for fold {fold_no} ...') 
        
        print("_" * 20) 
        print("lr, CoeffS, num_epochs, Drop_Rate, ii")
        print([lr, CoeffS, num_epochs, Drop_Rate, ii])
        print("_" * 20)
        DNN = train_DNN(DNN, X_train, y_train, X_test, y_test, callbacks=get_callbacks); #, myCallback
       
        y_score_te = predict_DNN(DNN, X_test, y_test); 
        #print('True test values and Predicted test values')
        X_Test_True_Pred = np.concatenate((y_test,y_score_te),axis=1)                     
        #print(X_Test_True_Pred)
        MSE_te[fold_no-1, ii] = mean_squared_error(y_test, y_score_te)        
        
        print(f'MSE Test Score for fold {fold_no} and L1 Regularized {CoeffS}: {MSE_te[fold_no-1, ii]}')
        MSE_per_fold.append(MSE_te[fold_no-1, ii])    
        del DNN
        fold_no = fold_no + 1         

end = time.time()
print(end - start)

CPU_Time = end - start    
print(MSE_te)
#Average_MSE = np.mean(MSE_te,axis=0)
Average_MSE = MSE_te.mean(axis=0) #column means
print(Average_MSE)

Opt_ind = np.argmin(Average_MSE)
print(Opt_ind)
Best_coeff = Param_Space['L1_Norm'].iat[Opt_ind]
Opt_epochs = int(Param_Space['Epoch'].iat[Opt_ind])
Best_LR = Param_Space['Learning_Rate'].iat[Opt_ind]
Best_Drop = Param_Space['Drop_Rate'].iat[Opt_ind]+0.2

print("_" * 20) 
print("Best_LR, Best_coeff, Opt_epochs, Best_Drop, Opt_ind")
print([Best_LR, Best_coeff, Opt_epochs, Best_Drop, Opt_ind])
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
        #if epoch+1 == num_epochs:
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
    model.fit(X, y, epochs=Opt_epochs, batch_size=batch_size, verbose=0, validation_split = validation_split, callbacks=[myCallback]);   
    return model;

DNN = getModelLocalEq(pVal, Best_coeff, Best_LR, Best_Drop);   # lrS[int(val)]
myCallback = My_Callback(outputDir, pVal);
DNN = train_DNN(DNN, x3D_all, Y, myCallback);

y_score = predict_DNN(DNN, x3D_all, Y); 
MSE_All = mean_squared_error(Y, y_score)       
print(MSE_All)

Grad2 = Gradients_All;

avg_all = np.zeros((pVal,Num_knock+1));
for row_ind in range(Grad2.shape[1]):
    for col_ind in range(Grad2.shape[2]):
        avg_all[row_ind,col_ind] = np.mean(Grad2[:,row_ind,col_ind]) 

Feat_Import = np.hstack((avg_all[:,0],avg_all[:,1],avg_all[:,2],avg_all[:,3],avg_all[:,4],avg_all[:,5]));

#Feat_Import_abs = np.abs(Feat_Import);

np.savetxt('FI_Reg_' + str(indx)+ '.csv', Feat_Import, delimiter=",")

METRICS_HMDP_New = [Opt_epochs, Best_coeff, Best_LR, Best_Drop, MSE_All, Average_MSE, np.min(Average_MSE), trainable_count,CPU_Time, indx]; 

with open(os.path.join(dataDir2, 'METRICS_HMDP_Reg_B.csv'), "a+") as fp:
    wr = csv.writer(fp, dialect='excel'); 
    wr.writerow(METRICS_HMDP_New);
try:
    os.stat(outputDir)
except:
    os.mkdir(outputDir)

print(METRICS_HMDP_New)
print('Done!')     


