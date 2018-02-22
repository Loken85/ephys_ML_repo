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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from collections import OrderedDict

# Random Forest Regression on single factors
# Trains and evaluates a RF regressor on predicting body position locations (one at a time) from
# time-binned spike counts of neurons
# Input: Matrices of binned spike counts, movement factors
# Outputs: Writes a ".mat" of test, validaiton, and training scores on the input set


# Set name of data set to load here
curr_data_set = '03_30_I_Blocks231_cat_data.mat'
out_file = '03_30_I_Blocks231_RF_MVMT_SFpreds.mat'

# Load in data set from .mat file
# Modify here to change the loaded matrices
def load_data_set():
    mat = scipy.io.loadmat(curr_data_set)
    #Stmtx = mat['STbintrim']
    Stmtx = mat['STbintrim']    
    Stmtx = np.transpose(Stmtx)
    #gridrefs = mat['gridrefs']
    #comdels = mat['comdels']
    #comtrim = mat['comtrim']
    #factors = mat['red_dists']
    factors = mat['red_dists']
    
    return Stmtx, factors
    
    
X, Y = load_data_set()

# Optional: Shorten X to match Y and remove the timestamp column

X = X[0:np.size(Y,0),1:]

# Optional: Standardize input matrix
X = StandardScaler().fit_transform(X)

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
    
# If you want a random data split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

# If you want a fixed data split
#X_train,Y_train,X_val,Y_val,X_test,Y_test = test_data_setup(X,Y) 

# Compute Mean squared error of regression predictions
# Inputs: y_true (true values), y_pred (predicted values)
# Output: mse (mean squared error), std (standard deviation of mse) 
def compute_mse(y_true, y_pred):
    se = np.square(y_true-y_pred)
    mse = np.mean(se)
    std = np.std(se)
    return mse, std 
    
# Build the Random Forest Estimator
# Adjust Build parameters here

# Initialize output matrices
scores=np.zeros((5,6))
test_preds=np.zeros((len(Y_test),6))
train_preds=np.zeros((len(Y_train),6))

for i in range(6):
    estimator  = RandomForestRegressor(n_estimators=1000,max_features='sqrt',oob_score='True')

# Run and Calculate Score
# Modify here to pull output for prediction checking
#score = cross_val_score(estimator, X, Y)

# Optional: split as time series
#tscv = TimeSeriesSplit(n_splits=5)

    score = cross_val_score(estimator,X_train,Y_train[:,i],cv=5)
    model = estimator.fit(X_train,Y_train[:,i])

# Print Session Name and Estimator
    print ("Random Forest Limb: %d Regression" % i)
    print(out_file)
    print("Cross Validated Scores")
    print("R2: %0.4f (+/- %0.4f)" % (score.mean(), score.std() * 2))
    print("OOB R2: %0.4f" %(estimator.oob_score_))
# Run predict on sets
    test_pred = estimator.predict(X_test)
    mse, std = compute_mse(Y_test[:,i], test_pred)
    print ("Results %.2f Test MSE" % (mse))


    train_pred = estimator.predict(X_train,)
    mse, std = compute_mse(Y_train[:,i], train_pred)
    print ("Results %.2f Train MSE" % (mse))
    
    scores[:,i] = score
    test_preds[:,i] = test_pred
    train_preds[:,i] = train_pred


# Assemble and write output: Saves to a ".mat" file using the out_file name  
out_dict = {'Y_test': Y_test, 'test_preds': test_preds, 'train_preds': train_preds, 'Y_train': Y_train, 'CV_scores': scores, 'OOB_score':(estimator.oob_score_) }
scipy.io.savemat(out_file,out_dict)



# Output Shuffle testing. Run the section below for shuffle control
#np.random.shuffle(Y)
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)
#estimator_s  = RandomForestRegressor(n_estimators=1000,max_features='sqrt',oob_score='True')
#model_s = estimator_s.fit(X_train,Y_train)
#test_preds = estimator_s.predict(X_test)
#mse, std = compute_mse(Y_test, test_preds)
#print ("Output Shuffle Results %.2f Test MSE" % (mse))
#train_preds = estimator_s.predict(X_train,)
#mse, std = compute_mse(Y_train, train_preds)
#print ("Output Shuffle Results %.2f Train MSE" % (mse))



