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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils.visualize_util import plot

# Pipeline for grid searching parameters for a CNN making predictions from
# binned spike counts
# This should be run on a selection of available data sets before selcting
# optimal parameters for the learning model for prediction


# Set name of data set to load here
curr_data_set = '04_07_Jay_Blocks132_cat_data.mat'

# Set out dimensions of the output data you will be training on
# Ex: 3 for gridrefs, 6 for movement factors, no. of categories etc.
out_dims = 6

# Load in data set from .mat file
# Modify here to change the loaded matrices
def load_data_set():
    mat = scipy.io.loadmat(curr_data_set)
    Stmtx = mat['STbintrim']
    Stmtx = np.transpose(Stmtx)    
    factors = mat['red_dists']
    return Stmtx, factors
    

# Function to define the learning model     
def learning_model(input_no, dropout_rate = 0.5, filter_width = 3, filter_no = 32, init_mode = 'normal'):
    model = Sequential()   
    
    
    
    # Convolution1d layer args: # filters, width of filter, activation, input shape
    # Input shape: # of samples (default none to allow any #), time_steps(this is the 
    # dimension along which the filters are applied), input dimensions (normally simply the dims
    # here we swap the "temporal" and space dimensions to facilitate the convolution), batch size
    model.add(Convolution1D(filter_no, filter_width, activation ='relu', input_shape = (input_no,1), init=init_mode))    
    model.add(MaxPooling1D())
    model.add(Convolution1D(filter_no, filter_width, activation= 'relu', init=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(filter_no,activation='relu',init=init_mode))
    model.add(Dense(out_dims, init=init_mode))
    
    
    
    
    # Select loss function and optimizer
    # These need be optimised as well, but this is best done separately
    # from the base parameter search
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    # Optional: Write diagram of model to file
    #plot(model, to_file='convNN.png')
    return model
    
  
# Process Data Block    
    
X, Y = load_data_set()

# Shorten X to match Y and remove the timestamp column

X = X[0:np.size(Y,0),1:]
      
# Optional: Standardize input matrix
X = StandardScaler().fit_transform(X)

# Optional to reduce dimensions on output      
#Y = Y[:,0:2]

# Optional for CNN usage, reshape input for temporal encoding
X = np.reshape(X,(-1,np.size(X,1),1))

# Setup Learning Block

#keras_estimators = []
# Optional: Can alternately be setup using sklearn's pipeline functionality
#keras_estimators.append(('standardize', StandardScaler()))
#keras_estimators.append(('mlp', KerasRegressor(build_fn=learning_model,nb_epoch=10, batch_size=32)))
#pipeline = Pipeline(keras_estimators)

# Initialize model
cnn_model = KerasRegressor(build_fn=learning_model,nb_epoch=100, input_no = np.size(X,1))

# Setup grid parameters
# Provided is an example of a reasonable grid for search
# Keep in mind that grid search is O=n^2, with n being the number of parameters
# This may take a very long time depending on hardware, and is not recommended
# if you are not using a GPU for parallelisation
batch_size = [10,32,64,96]
dropout_rate = [0.1,0.2,0.3,0.4,0.5,0.6]
filter_width = [3,5,7,11,15]
filter_no = [10,32,64,96]
#init_mode = ['uniform','normal','zero']

# Setup grid
param_grid = dict(batch_size=batch_size, dropout_rate=dropout_rate, filter_width = filter_width, filter_no = filter_no)
grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid,scoring='neg_mean_squared_error',verbose=2)
# Alternate scoring examples
#grid = GridSearchCV(estimator=cnn_model, param_grid=param_grid,scoring='r2')


# Run grid
grid_result = grid.fit(X,Y)

# print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

   
    

