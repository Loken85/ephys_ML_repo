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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM
from keras.utils.visualize_util import plot

# Pipeline for grid searching parameters for an RNN making predictions from
# binned spike counts
# This should be run on a selection of available data sets before selcting
# optimal parameters for the learning model for prediction

# Set name of data set to load here
curr_data_set = '04_07_Jay_Blocks132_data.mat'

# Load in data set from .mat file
# Modify here to change the loaded matrices
def load_data_set():
    mat = scipy.io.loadmat(curr_data_set)
    Stmtx = mat['STbintrim']
    Stmtx = np.transpose(Stmtx)
    gridrefs = mat['gridrefs']
    
    
    return Stmtx, gridrefs
    

# Function to define the learning model 
def learning_model(input_no, dropout_rate = 0.2, steps = 3, lag=0, units = 32, init_mode='uniform'):
    model = Sequential()
    
    
        
    
    
    
    model.add(LSTM(units,return_sequences=True, input_shape=(input_no,steps), init=init_mode))
    model.add(LSTM(units,return_sequences=True, init=init_mode))
    model.add(LSTM(units, init=init_mode))
    model.add(Dense(3))
    
    
    
    
    # Select loss function and optimizer
    # These need be optimised as well, but this is best done separately
    # from the base parameter search    
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    return model
    
  
# Process Data Block    
    
X, Y = load_data_set()

# Shorten X to match Y and remove the timestamp column

X = X[0:np.size(Y,0),1:]

# Optional to reduce dimensions on output      
#Y = Y[:,0:2]

# Optional: Standardize input matrix
X = StandardScaler().fit_transform(X)

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
steps = 3
lag = 0
    
# Call to reshape for RNN, with steps and lag
X, Y = temporal_reshape(X, Y, steps, lag)

# Setup Learning Block

# Optional: Can alternately be setup using sklearn's pipeline functionality
#keras_estimators.append(('standardize', StandardScaler()))
#keras_estimators.append(('mlp', KerasRegressor(build_fn=learning_model,nb_epoch=10, batch_size=32)))
#pipeline = Pipeline(keras_estimators)

# Initialize model
dense_model = KerasRegressor(build_fn=learning_model,nb_epoch=10, input_no = np.size(X,1), steps=steps, lag=lag)

# Setup grid parameters
# Provided is an example of a reasonable grid for search
# Keep in mind that grid search is O=n^2, with n being the number of parameters
# This may take a very long time depending on hardware, and is not recommended
# if you are not using a GPU for parallelisation
batch_size = [10,32,64,96]
dropout_rate = [0.1,0.2,0.3,0.4,0.5,0.6]
units = [10,32,64,96]
init_mode = ['uniform','normal','zero']

# Setup grid
# Feel free to experiment with other scoring functions
param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, units=units, init_mode=init_mode)
grid = GridSearchCV(estimator=dense_model, param_grid=param_grid,scoring='neg_mean_squared_error')


# Run grid
grid_result = grid.fit(X,Y)

# print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




