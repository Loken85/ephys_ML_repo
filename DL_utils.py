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

import matplotlib.pyplot as plt
from collections import OrderedDict


# COLLECTION OF UTILITY FUNCTIONS

# Function to replace individual neuron spike trains with random poisson RVs 
# with matching lambdas
# Input: stmtx (matrix of spike trains, bins x neurons)
# Output: swap_stmtxs (list of stmtx matrices each with one neuron replaced)
# And one additional matrix with all neurons replaced, and original matrix
def poisson_replace(stmtx):
    swap_stmtxs = []
    bins = np.size(stmtx,0)
    swap_stmtxs.append(stmtx)
    all_swap = np.copy(stmtx)
    for i, col in enumerate(np.transpose(stmtx)):
        lb = np.mean(col)
        prv = np.random.poisson(lb,bins)
        swapped = np.copy(stmtx)
        swapped[:,i] = prv
        all_swap[:,i] = prv
        swap_stmtxs.append(swapped)
        
    swap_stmtxs.append(all_swap)
    return swap_stmtxs
    
    
# Compute accuracy of categorical predictions
# Inputs: y_true (categorical labels), y_pred(categorical probabilites)
# Output: acc (accuracy of highest likelihood category vs true category)
def compute_acc(y_true, y_pred):
    acc = np.mean(np.equal(np.argmax(y_true,axis=-1),np.argmax(y_pred,axis=-1)))
    return acc
    

# Compute Mean squared error of regression predictions
# Inputs: y_true (true values), y_pred (predicted values)
# Output: mse (mean squared error), std (standard deviation of mse)
def compute_mse(y_true, y_pred):
    se = np.square(y_true-y_pred,axis=-1)
    mse = np.mean(se)
    std = np.std(se)
    return mse, std
    

 
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
    
    
    
# function for testing of poisson replace across all neurons
# 
def neuron_replace_test(X,Y,model):

    stmtxs = poisson_replace(X)
    for i, mat in enumerate(stmtxs):
        labels = np.copy(Y)
        stmtx = stmtxs[i]
        # Scaler if using CNN
        stmtx = StandardScaler().fit_transform(stmtx)
        stmtx, labels = temporal_reshape(stmtx,labels,steps,lag) 
        labels = to_categorical(labels)
        X_train,Y_train,X_val,Y_val,X_test,Y_test = test_data_setup(stmtx,labels)
        print ("Neuron: %.0f" % (i))
        predictions = model.predict(X_test,batch_size=10,verbose=0)
        acc = np.mean(np.equal(np.argmax(Y_test,axis=-1),np.argmax(predictions,axis=-1)))
        print ("Results %.2f Test Accuracy" % (acc))

        predictions = model.predict(X_val,batch_size=10,verbose=0)
        acc = np.mean(np.equal(np.argmax(Y_val,axis=-1),np.argmax(predictions,axis=-1)))
        print ("Results %.2f Validation Accuracy" % (acc))

        predictions = model.predict(X_train,batch_size=10,verbose=0)
        acc = np.mean(np.equal(np.argmax(Y_train,axis=-1),np.argmax(predictions,axis=-1)))
        print ("Results %.2f Train Accuracy" % (acc))
        # TODO: Write results into matrix for file out
        
        
# Calculate a reliability measure for the categorisation
def rel_measure(X_train,X_test,Y_train,test_preds):
    # Find categories
    cats = np.unique(Y_train)
    mses = np.zeros((np.size(cats,0),1))
    # For each category
    for i, cat in enumerate(cats):
        if (np.ndim(X_train)==2):
            # Get average input for the category
            inds = np.where(Y_train==cat)
            ins = X_train[inds[0],:]
            avg = np.mean(ins,0)
            # Get all test inputs that predict the category
            inds = np.where(test_preds==cat)
            ins = X_test[inds[0],:]
            # if there are no instances of this cat, insert zero row for max error        
            if (np.size(ins)==0):
                ins = np.zeros((1,np.size(X_train,1)))
                # Calc MSE for the category: subtract, square, then sum across rows &
                # columns
            tot_err = np.sum(np.square(np.subtract(ins,avg)))
            mses[i] = tot_err/(np.size(ins,0)*np.size(ins,1))
                
        elif (np.ndim(X_train)==3):
            # Get average input for the category
            inds = np.where(Y_train==cat)
            ins = X_train[inds[0],:,:]
            avg = np.mean(ins,0)
            # Get all test inputs that predict the category
            inds = np.where(test_preds==cat)
            ins = X_test[inds[0],:,:]
            # if there are no instances of this cat, insert zero row for max error        
            if (np.size(ins)==0):
                ins = np.zeros((1,np.size(X_train,1),np.size(X_train,2)))
                # Calc MSE for the category: subtract, square, then sum across rows &
                # columns
            tot_err = np.sum(np.square(np.subtract(ins,avg)))
            mses[i] = tot_err/(np.size(ins,0)*np.size(ins,1)*np.size(ins,2))
    
    rel = np.sum(mses)/(np.size(cats,0))
    return rel
    
    
# Uncategoricals a categorical array
def un_categorical(cat_arr):
    categories = np.arange(0,np.size(cat_arr,1),1)
    return categories[cat_arr.argmax(1)]
                      
                      
# Example REL testing
#y_train = un_categorical(Y_train)
#rel = rel_measure(X_train,X_test,y_train,test_preds)
#print("REL:%.2f" % (rel))
                      

 # Out-of-bin testing function for RF regression. Evaluates and plots OOB error over changing parameters                     
def prog_OOB_test_r(X,y):

    RANDOM_STATE = 123

    ensemble_clfs = [
    ("RandomForestRegressor, max_features='sqrt'",
        RandomForestRegressor(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features='log2'",
        RandomForestRegressor(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestRegressor, max_features=None",
        RandomForestRegressor(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
    ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 100
    max_estimators = 1000

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 50):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = clf.oob_score_
            error_rate[label].append((i, oob_error))

            # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB R2")
    plt.legend(loc="lower right")
    plt.show()


    
# Out-of-bin testing function for RF classifier
def prog_OOB_test_c(X,y):

    RANDOM_STATE = 123

    ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
    ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 100
    max_estimators = 1000

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1, 50):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

            # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()




    
    
