# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:25:27 2020

@author: cleml
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import sklearn.linear_model as lm
from Functions import rlr_validate, setupI_reg
from Functions import train_neural_net, draw_neural_net
import torch

#REMEMBER TO USE WHOLE DATA WHEN CODE RUNS

# Loading data
filename = 'heart.csv'
df = pd.read_csv(filename)

# Making one-of-K encoding of categorical data
a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
d = pd.get_dummies(df['restecg'], prefix = "restecg")

# Extracting target subdataframe from data frame
targ = df.pop('target')

# Concatenating current data frame and encoded one-of-k values and
# making sure target is the  last attribute of the data frame
frames = [df, a, b, c, d, targ]
df = pd.concat(frames, axis = 1)

# Dropping original form of one-of-K encoded variables from dataframe
df = df.drop(columns = ['cp', 'thal', 'slope', 'restecg'])

# Getting attributes names
attributeNames = list(df.columns)
attNb = len(attributeNames)

# splitting dataframe into X and y
X = df.drop(['thalach'], axis = 1).to_numpy()
y = df[['thalach']].to_numpy().reshape((len(X),1))
#y = data[:,len(data[0,:])-1].reshape((len(X),1))

K1 = 10     # Outer-crossvalidation fold
K2 = 10     # inner-crossvalidation fold

CV = model_selection.KFold(K1, shuffle = True, random_state = 12)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Array to store optimal weights of multiple linear regression for optimal lambdas 
wopt = np.empty((X.shape[1]+1,K1))

opt_lambda = np.empty(K1)
opt_n_hidden_units = np.empty(K1)
# Array to store error of linear regression for all K1 folds
lr_test_error = np.empty(K1)
# Array to store error of linear regression for all K1 folds
ann_test_error = np.empty(K1)
# Array to store error of baseline model for all K1 folds
baseline_test_error = np.empty(K1)

Lst_Baseline_test = []
Lst_Mlr_test = []
Lst_ANN_test = []
Lst_True_test = []

# Outer-crossvalidation loop - used to compute generalisation error for all models
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    
    print('\Outer-Crossvalidation fold: {0}/{1}'.format(k+1,K1)) 
    
    X_train = X[train_index]
    X_test = X[test_index]
    
    y_train = y[train_index]
    y_test = y[test_index]
    
    N,M = X_train.shape
    
    CV2 = model_selection.KFold(K2, shuffle = True, random_state = 12)
    f = 0
    
    #Defining empty weight array for the weights corresponding to the M
    #attributes as a function of the fold and the value of lambda
    w = np.empty((M+1,K2,len(lambdas)))
    train_error_lr = np.empty((K2,len(lambdas)))
    test_error_lr = np.empty((K2,len(lambdas)))
    
    # Parameters for neural network classifier
    # 3 hidden units was found to be optimal
    #n_hidden_units = [1]
    n_hidden_units = np.arange(1,5,1)      #range of numbers of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 100000        # number of iterations of gradient descent
    # Array to store average test error of each trained neural network 
    mse = np.empty((K2, len(n_hidden_units)))
    
    for (j,(train_idx, test_idx)) in enumerate(CV2.split(X_train,y_train)):
        
        print('\Inner-Crossvalidation fold: {0}/{1}'.format(j+1,K1)) 
        
        X_p_train = X_train[train_idx]
        X_p_test = X_train[test_idx]
        
        y_p_train = y_train[train_idx]
        y_p_test = y_train[test_idx]
        
        N_in = X_p_train.shape[0]
        
        # Standardize the training and test set based on training set moments
        mu_in = np.mean(X_p_train, 0)
        sigma_in = np.std(X_p_train, 0)
        
        # Check if not all values of an attribute are 0, to avoid division by 0
        for i in np.where(mu_in==0)[0]:
            sigma_in[i] = 1
        
        # Standardizing the data
        X_p_train = (X_p_train - mu_in) / sigma_in
        X_p_test = (X_p_test - mu_in) / sigma_in
        
        #--------------- Linear regression part -------------------------------
        
        # Concatenating 1 to account for offset (constant term) of linear regression
        X_p_train_reg = np.concatenate((np.ones((N_in,1)),X_p_train),1)
        X_p_test_reg = np.concatenate((np.ones((len(X_p_test),1)),X_p_test),1)
        
        # Precomputing matrix multiplications
        Xty_p = X_p_train_reg.T @ y_p_train
        XtX_p = X_p_train_reg.T @ X_p_train_reg
        
        # Computing coefficients of multiple linear regression for different
        # values of lambda
        for l in range(0,len(lambdas)):
            
            # The matrix lambda*I of the regression with regularization
            lambdaI = lambdas[l] * np.eye(M+1)
            lambdaI[0,0] = 0 # remove bias regularization
            
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            w[:,f,l] = np.linalg.solve(XtX_p+lambdaI,Xty_p).squeeze()
            
            # Evaluating +0training and test performance
            train_error_lr[f,l] = np.power(y_p_train.squeeze()-X_p_train_reg @ w[:,f,l].T,2).mean(axis=0)
            test_error_lr[f,l] = np.power(y_p_test.squeeze()-X_p_test_reg @ w[:,f,l].T,2).mean(axis=0)
    
        #--------------- ANN part ---------------------------------------------
        
        # Converting arrays to tensors for pytorch
        X_p_train_ann = torch.Tensor(X_p_train)
        y_p_train_ann = torch.Tensor(y_p_train)
        X_p_test_ann = torch.Tensor(X_p_test)
        y_p_test_ann = torch.Tensor(y_p_test)
        
        # Training neural networks for various numbers of hidden units
        for i in range(0,len(n_hidden_units)):
            
            # Defining structure of neural network
            
            model = lambda: torch.nn.Sequential(
                        # Test for comparison with mult linear regression                
                        #torch.nn.Linear(M, 1), #M features to n_hidden_units
                        torch.nn.Linear(M, n_hidden_units[i]), #M features to n_hidden_units
                        #torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden_units[i], 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_p_train_ann,
                                                       y=y_p_train_ann,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
            print('\n\tBest loss: {}\n'.format(final_loss))
    
            # Determine estimated class labels for test set
            y_test_est = net(X_p_test_ann)
        
            # Determine errors and errors
            se = (y_test_est.float()-y_p_test_ann.float())**2 # squared error
            mse[f,i] = (sum(se).type(torch.float)/len(y_p_test)).data.numpy() #mean
            
        f=f+1
    # Getting optimal lambda i.e. the one yielding the smallest test error
    opt_val_err = np.min(np.mean(test_error_lr,axis=0))
    opt_lambda[k] = lambdas[np.argmin(np.mean(test_error_lr,axis=0))]
    
    # Getting optimal number of hidden units
    opt_n_hidden_units[k] = n_hidden_units[np.argmin(np.mean(mse,axis=0))]
    # Trick to get arround pytorch typeyt error (just ignore)
    opt_h_un = n_hidden_units[np.argmin(np.mean(mse,axis=0))]
    # Standardize the training and set based on training set moments
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
        
    # Check if not all values of an attribute are 0, to avoid division by 0
    for i in np.where(mu == 0)[0]:
        sigma[i] = 1
        
    # Standardizing the data
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    
    # -------------------- Multiple linear regression ------------------------
    
    # Concatenating 1 to account for offset (constant term) of linear regression
    X_train_reg = np.concatenate((np.ones((N,1)),X_train),1)
    X_test_reg = np.concatenate((np.ones((len(X_test),1)),X_test),1)
    
    # Doing matrix multiplications for later computations
    Xty = X_train_reg.T @ y_train
    XtX = X_train_reg.T @ X_train_reg
    
    # Computing the matrix lambda*I of the regression with regularization
    lambdaI = opt_lambda[k] * np.eye(M+1)
    lambdaI[0,0] = 0 # remove bias regularization
    
    # Computing optimal coefficients of multiple linear regression with 
    # regularization for optimal lambda
    wopt[:,k] = np.linalg.solve(XtX + lambdaI,Xty).squeeze()
    
    # Compution test error of multiple linear regression for optimal lambda
    lr_test_error[k] = np.power(y_test.squeeze()-X_test_reg@ wopt[:,k].T,2).mean(axis=0)
    
    # ----------------------- ANN --------------------------------------------
    
    # Defining structure of neural network with optimal number of hidden units
    model = lambda: torch.nn.Sequential(
                        # Test for comparison with mult linear regression                
                        #torch.nn.Linear(M, 1), #M features to n_hidden_units
                        torch.nn.Linear(M, opt_h_un), #M features to n_hidden_units
                        #torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.ReLU(),
                        torch.nn.Linear(opt_h_un, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    # Converting numpy arrays to tensors for py.torch
    X_train_ann = torch.Tensor(X_train)
    y_train_ann = torch.Tensor(y_train)
    X_test_ann = torch.Tensor(X_test)
    y_test_ann = torch.Tensor(y_test)
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X = X_train_ann,
                                                       y = y_train_ann,
                                                       n_replicates = n_replicates,
                                                       max_iter=max_iter)
    # Computing estimate for y_test using trained neural network with optimal
    # number of hidden units
    y_test_est = net(X_test_ann)
        
    # Computing test error for neural network
    se = (y_test_est.float()-y_test_ann.float())**2 # squared error
    ann_test_error[k] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    # ---------------------- Baseline Error ----------------------------------
    
    y_hat = np.mean(y_train)
    
    baseline_test_error[k] = np.square(y_test-y_hat).mean(axis=0)
    
    Lst_Mlr_test.append(X_test_reg@ wopt[:,k])
    Lst_Baseline_test.append(y_hat)
    Lst_ANN_test.append(y_test_est)
    Lst_True_test.append(y_test)
    
    
plt.figure()
plt.plot(np.arange(1,K1+1,1),ann_test_error,'x-')
plt.plot(np.arange(1,K1+1,1),lr_test_error,'o-')
plt.plot(np.arange(1,K1+1,1),baseline_test_error,'k--')
plt.xlabel('Fold number',fontsize = 12)
plt.ylabel('Mean square test error', fontsize = 12)
plt.xticks(np.arange(1,K1+1,1),np.arange(1,K1+1,1))
plt.legend(['ANN','Mul. lin reg.','baseline'])
plt.tight_layout()
plt.savefig('C:/Users/cleml/Documents/02450 Introduction to Machine Learning and Data Mining/Project 2/test.pdf',bbox_inches='tight')
plt.show()
    
#np.savetxt('ANN_data.csv', [[f'y_test_ann_{k+1}',p.detach().numpy()] for (k,p) in enumerate(Lst_ANN_test)] , delimiter=',', fmt='%s')  

# Concatenating y_test and estimated y_test data for the K1 folds into 1 array.

# Concatenating array from lists of arrays
Y_True_test = np.concatenate(Lst_True_test, axis=0)

# Concatening tensor from list of tensors and converting it to numpy
Y_ANN_test = torch.cat(Lst_ANN_test).detach().numpy()

# Concatenating array from list of arrays and giving same shape as the others
Y_Mlr_test = np.concatenate(Lst_Mlr_test, axis=0).reshape(-1,1)

# Taking every single element of the baseline and duplicating it to match the
# number of elements in the other arrays + concatenating and reshaping it
Y_Baseline_test = np.concatenate([np.repeat(i,len(Lst_True_test[k]))
                                  for (k,i) in enumerate(Lst_Baseline_test)],
                                 axis = 0).reshape(-1,1)


CIA, CIB, CI, p = setupI_reg(Y_True_test,Y_Mlr_test,Y_ANN_test, L1_Loss = True, alpha = 0.05)
print('For the Multiple linear regression vs the ANN we get: \n \n'
      f'CI y_true vs y_Mlr_est = {CIA}\n'
      f'CI y_true vs y_ANN_est = {CIB}\n'
      f'CI Mlr vs ANN = {CI}\n'
      f'p-value = {p}\n')
print('\n')

CIA, CIB, CI, p = setupI_reg(Y_True_test,Y_Mlr_test,Y_Baseline_test, L1_Loss = True, alpha = 0.05)
print('For the Multiple linear regression vs the ANN we get: \n \n'
      f'CI y_true vs y_Mlr_est = {CIA}\n'
      f'CI y_true vs y_Baseline_est = {CIB}\n'
      f'CI Mlr vs Baseline = {CI}\n'
      f'p-value = {p}')

CIA, CIB, CI, p = setupI_reg(Y_True_test,Y_ANN_test,Y_Baseline_test, L1_Loss = True, alpha = 0.05)
print('For the Multiple linear regression vs the ANN we get: \n \n'
      f'CI y_true vs y_ANN_est = {CIA}\n'
      f'CI y_true vs y_Baseline_est = {CIB}\n'
      f'CI ANN vs Baseline = {CI}\n'
      f'p-value = {p}')

   
  #REMEMBER STANDARDIZATION OF DATA IN OUTER FOLD
    
    #VERIFY WAY THE ERRORS ARE DEFINED
    
    # RECHANGE NEURAL NETWORK STRUCTURE
    # IN FUNCTIONS UNCOMMENT:
    #torch.nn.init.xavier_uniform_(net[2].weight)
    # ARTIFICIAL NEURAL NETWORK PERFORMS ALMOST AS WELL AS MUL LIN REGRESSION
    # WHEN ONLY ONE LAYERS WITH LINEAR WEIGHTS
    
        