# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 08:04:33 2020

@author: mbust
"""

#In this script we are doing the 2-level cross-validation to create 
#a Table comparing the models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show, boxplot
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from Functions import rlr_validate

"""
--------------------------------------------
-------PHASE 1 - GETTING DATA READY---------
--------------------------------------------
"""
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
X = df.drop(['target'], axis = 1).to_numpy()
y= targ.to_numpy()
N,M = X.shape

"""
-------------------------------------------------
-------PHASE 2 - SET UP CROSS-VALIDATION---------
-------------------------------------------------
"""

K1 = 10     # Outer-crossvalidation fold
K2 = 10     # inner-crossvalidation fold

CV = model_selection.KFold(K1, shuffle = True, random_state = 12)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Array to store optimal weights of multiple linear regression for optimal lambdas 
wopt = np.empty((X.shape[1]+1,K1))

opt_lambda = np.empty(K1)
opt_tc_units = np.empty(K1)
# Array to store error of logistic regression for all K1 folds
lr_test_error = np.empty(K1)
# Array to store error of classification trees for all K1 folds
cltr_test_error = np.empty(K1)
# Array to store error of baseline model for all K1 folds
baseline_test_error = np.empty(K1)

Lst_Baseline_test = []
Lst_ClTree_test = []
Lst_lr_test = []

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
        for i in np.where(sigma_in==0)[0]:
            sigma_in[i] = 1
        
        # Standardizing the data
        X_p_train = (X_p_train - mu_in) / sigma_in
        X_p_test = (X_p_test - mu_in) / sigma_in
        
        """
        --------------------------------------------
        -------PHASE 3 - Classification tree---------
        --------------------------------------------
        """
        # Parameters for tree classifier
        # 3 complexity levels was found to be optimal
        #tree depth complexity
        tc = 3
        
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=tc)
        dtc = dtc.fit(X_p_train,y_p_train.ravel()) #is sometimes problematic (NaN input)
        y_est_test = dtc.predict(X_p_test)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_p_test) / float(len(y_est_test))
        cltr_test_error[j] = misclass_rate_test
        
        
        """
        --------------------------------------------
        -------PHASE 4 - LOGISTIC REGRESSION--------
        --------------------------------------------
        """
            


        # Fit regularized logistic regression model to training data to predict 
        # the type of wine
        lambda_interval = np.logspace(-8, 2, 50)
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        coefficient_norm = np.zeros(len(lambda_interval))
        opt_lambda = 
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
            
            mdl.fit(X_train, y_train)
        
            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T
            
            train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
            test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
        
            w_est = mdl.coef_[0] 
            coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
        
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        """
        # Concatenating 1 to account for offset (constant term) of logistic regression
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
    """
    
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
"""
--------------------------------------------
-------PHASE 5 - BASELINE---------
--------------------------------------------
"""
    largestClass = np.argmax(np.array([len(y_train[y_train==0]),len(y_train[y_train==1])]))
    
    misclass_rate_test = sum(largestClass != y_test.squeeze()) / (len(y_test.squeeze()))
    baseline_test_error[k] = misclass_rate_test
    
"""
-----------------------------------------------------------
-------PHASE 6 - STORING ERRORS + PLOTTING RESULTS---------
-----------------------------------------------------------
"""
 
    Lst_Baseline_test.append(baseline_test_error[k])
    Lst_ClTree_test.append(np.mean(cltr_test_error))
    Lst_lr_test.append(X_test_reg@ wopt[:,k])
    
plt.figure()
plt.plot(np.arange(1,K1+1,1),cltr_test_error,'x-')
plt.plot(np.arange(1,K1+1,1),lr_test_error,'o-')
plt.plot(np.arange(1,K1+1,1),baseline_test_error,'k--')
plt.xlabel('Fold number',fontsize = 12)
plt.ylabel('Mean square test error', fontsize = 12)
plt.xticks(np.arange(1,K1+1,1),np.arange(1,K1+1,1))
plt.legend(['Class. Tree','log. reg.','baseline'])
plt.tight_layout()
plt.show()