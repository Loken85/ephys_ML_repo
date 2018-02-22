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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt
from collections import OrderedDict


# Random Forest Classification on place
# Trains and evaluates a RF classifier on locations from
# time-binned spike counts of neurons
# Input: Matrices of binned spike counts, locations as categories
# Outputs: Writes a ".mat" of test, validaiton, and training scores on the input set


# Set name of data set to load here
curr_data_set = '04_02_I_Blocks132_cat_data.mat'
out_file = '04_02_I_Blocks132_RF_place_preds.mat'

#curr_data_set = '315-1_baseline_hipp_data.mat'
#out_file = '315-1_baseline_RF_cat_preds.mat'

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
    gridints = mat['gridints']
    
    return Stmtx, gridints

# Optional: Load data function for dividing the session into blocks    
def load_data_set_div():
    mat = scipy.io.loadmat(curr_data_set)
    #Stmtx = mat['STbintrim']
    Stmtx = mat['STbin_sout']    
    X_train = np.transpose(Stmtx)
    X_test = np.transpose(mat['STbin_shock_nout'])
    #gridrefs = mat['gridrefs']
    #comdels = mat['comdels']
    #comtrim = mat['comtrim']
    #factors = mat['red_dists']
    Y_train = mat['gridints_sout']
    Y_test = mat['gridints_shock_nout']
    
    return X_train,X_test,Y_train,Y_test
    
# Standard data load   
X, Y = load_data_set()

# Predivided data load
#X_train,X_test,Y_train,Y_test = load_data_set_div();

# Shorten X to match Y and remove the timestamp column

X = X[0:np.size(Y,0),1:]
#X_train = X_train[0:np.size(Y_train,0),1:]
#X_test = X_test[0:np.size(Y_test,0),1:]

# Optional: Standardize input matrix
X = StandardScaler().fit_transform(X)
#X_train = StandardScaler().fit_transform(X_train)
#X_test = StandardScaler().fit_transform(X_test)

#Y = to_categorical(Y)

# Random data split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

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
    
# Fixed data split
#X_train,Y_train,X_val,Y_val,X_test,Y_test = test_data_setup(X,Y) 



# Build the Random Forest Estimator
# Adjust Build parameters here
estimator  = RandomForestClassifier(n_estimators=1000, max_features='log2', oob_score='True', class_weight='balanced')

# Run and Calculate Score
# Modify here to pull output for prediction checking
#score = cross_val_score(estimator, X, Y)

# Optional: split as time-series
#tscv = TimeSeriesSplit(n_splits=5)

scores = cross_val_score(estimator,X_train,np.ravel(Y_train),cv=5)
model = estimator.fit(X_train,np.ravel(Y_train))



# Print Session Name and Estimator
print ("Random Forest Place Classification")
print(out_file)
print("Cross Validated Scores")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("Single Run Predictions")
# Run predict on sets
test_preds = estimator.predict(X_test)
acc = estimator.score(X_test,Y_test)
print ("Results %.2f Test Accuracy" % (acc))


train_preds = estimator.predict(X_train)
acc = estimator.score(X_train,Y_train)
print ("Results %.2f Train Accuracy" % (acc))


# Assemble and write output: Saves to a ".mat" file using the out_file name 
out_dict = {'Y_test': Y_test, 'test_preds': test_preds, 'train_preds': train_preds, 'Y_train': Y_train, 'CV_scores': scores}
scipy.io.savemat(out_file,out_dict)



