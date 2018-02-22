# -*- coding: utf-8 -*-

# Code for analysis presented in 
#'How much does movement and location encoding impact prefrontal cortex activity? 
# An algorithmic decoding approach in freely moving rats.'
#
# by Adrian Lindsay, Barak Caracheo, Jamie Grewal, Daniel Leibovitz, and Jeremy Seamans
#     Author: Adrian Lindsay
#     E-mail: adrianj.lindsay@gmail.com
#       Date: January 15, 2018
#Institution: University of British Columbia
#Copyright (C) 2018 Adrian Lindsay
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#'''



import scipy.io
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM
from keras.utils.visualize_util import plot


# Recurrent Neural Network regressor on movement factors
# Trains and evaluates a RNN on movement factor values from time-binned spike counts
# Input: Matrices of binned spike counts, movement factors
# Outputs: Writes a ".mat" of test, validaiton, and training scores on the input set
 
# Set name of data set to load here
curr_data_set = '09_22_L_base_cat_data.mat'
out_file = '09_22_L_base_RNN_MVMT_preds.mat'

# Set out dimensions of the output data you will be training on
# Ex: 3 for gridrefs, 6 for movement factors, no. of categories etc.
out_dims = 6

# Load in data set from .mat file
# Modify here to change the loaded matrices
def load_data_set():
    mat = scipy.io.loadmat(curr_data_set)
    Stmtx = mat['STbintrim']
    Stmtx = np.transpose(Stmtx)
    #gridrefs = mat['gridrefs']
    #comdels = mat['comdels']
    #comtrim = mat['comtrim']
    factors = mat['red_dists']
    
    return Stmtx, factors
    

# Function to define the learning model    
# Arguments as provided are optimised for movement factor prediction   
def learning_model(input_no, dropout_rate = 0.5, steps = 3, lag=2, units = 32, init_mode='normal'):
    model = Sequential()
    
       
    
    
    model.add(LSTM(units,return_sequences=True, input_shape=(input_no,steps),init=init_mode))
    model.add(LSTM(units,return_sequences=True,init=init_mode))
    model.add(LSTM(units,init=init_mode))
    model.add(Dense(out_dims))
    
    
    
    
    # Select loss function and optimizer
   
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    # Optional: plot a diagram of the model
    #plot(model, to_file='RNN.png')
    return model
    
  
# Process Data Block    
    
X, Y = load_data_set()

# Shorten X to match Y and remove the timestamp column

X = X[0:np.size(Y,0),1:]

# Optional: Standardize input matrix
X = StandardScaler().fit_transform(X)

# Optional to reduce dimensions on output      
#Y = Y[:,0:2]

# For RNN usage, reshape input for temporal encoding with variable timesteps
# Steps is the number of bins to be included in one input
# Lag is the offset into the labels bins to use as the sequence label
# Lag must be < steps
def temporal_reshape(data, labels, steps, lag):
    
    # trim arrays to compatible length for steps
    sample_trim = np.size(data,0)//steps
    data = data[0:(sample_trim*steps),:]
    labels = labels[0:(sample_trim*steps),:]
    
    data = np.reshape(data,(-1,np.size(data,1),steps))
    labels = np.reshape(labels,(-1,np.size(labels,1),steps))
    
    # select the sequence label based on the lag parameter
    labels = labels[:,:,lag]
    labels = np.reshape(labels,(-1,np.size(labels,1)))
    
    
    return data, labels

# Search through steps and lag params once grid-search of parameters is complete
# Example: Use steps 1,3,5
# Lag 0,0:1:2,0:1:2:3:4)
steps = 5
lag = 3
    
# Call to reshape for RNN, with steps and lag
X, Y = temporal_reshape(X, Y, steps, lag)


# Function to setup test and validation data by divying up data set
# Input: X (input matrix), Y (output vector/matrix)
# Output: X_train, X_test, X_val (training, test, and validation sets for input)
# Y_train, Y_test, Y_val (training, test, and validation sets for output)
def test_data_setup(X, Y):
    length = np.size(X,0)
    train_size = np.floor(length/10)*9    
    X_test = X[train_size:length,:]
    Y_test = Y[train_size:length,:]
    val_size = np.floor(train_size/10)
    X_train = X[0:(train_size-val_size),:]
    Y_train = Y[0:(train_size-val_size),:]
    X_val = X[(train_size-val_size):train_size,:]
    Y_val = Y[(train_size-val_size):train_size,:]
    
    return X_train,Y_train,X_val,Y_val,X_test,Y_test
    
# Random data split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

# Fixed data split
X_train,Y_train,X_val,Y_val,X_test,Y_test = test_data_setup(X,Y)

# Compute Mean squared error of regression predictions
# Inputs: y_true (true values), y_pred (predicted values)
# Output: mse (mean squared error), std (standard deviation of mse) 
def compute_mse(y_true, y_pred):
    se = np.square(y_true-y_pred)
    mse = np.mean(se)
    std = np.std(se)
    return mse, std 


# Setup Learning Block


model = learning_model(input_no = np.size(X,1), steps=steps, lag=lag)

# Run fit on learning model
#model.fit(X_train, Y_train, batch_size=96,nb_epoch=2000,verbose=1)

# Optional: manual validation version
model.fit(X_train,Y_train,batch_size=96,nb_epoch=2000,verbose=1,validation_data=(X_val,Y_val))


# Run predict on sets
test_preds = model.predict(X_test,batch_size=10,verbose=1)
mse, std = compute_mse(Y_test, test_preds)
print ("Results %.2f Test MSE" % (mse))

# Rerun validation predictions
val_preds = model.predict(X_val,batch_size=10,verbose=1)
mse, std = compute_mse(Y_val, val_preds)
print ("Results %.2f Validation MSE" % (mse))

# Rerun training predictions
train_preds = model.predict(X_train,batch_size=10,verbose=1)
mse, std = compute_mse(Y_train, train_preds)
print ("Results %.2f Train MSE" % (mse))


# Assemble and write output: Saves to a ".mat" file using the out_file name 
out_dict = {'Y_test': Y_test, 'test_preds': test_preds, 'val_preds': val_preds, 'train_preds': train_preds, 'Y_val': Y_val, 'Y_train': Y_train}
scipy.io.savemat(out_file,out_dict)

