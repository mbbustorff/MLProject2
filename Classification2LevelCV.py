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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

CV = model_selection.KFold(K1, shuffle = True, random_state = 121)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Array to store optimal weights of multiple linear regression for optimal lambdas 
wopt = np.empty((X.shape[1]+1,K1))

opt_lambda = np.empty(K1)
opt_tc_units = np.empty(K1)
# Array to store error of logistic regression for all K1 folds
lr_test_error = np.empty((K1))
# Array to store error of classification trees for all K1 folds
cltr_test_error = np.empty((K1))

# Array to store error of baseline model for all K1 folds
baseline_test_error = np.empty(K1)


Lst_ClTree_test = []
Lst_tc = []
Lst_lr_test = []
Lst_lambdas = []
Lst_Baseline_test = []

# Outer-crossvalidation loop - used to compute generalisation error for all models
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
    
    print('\Outer-Crossvalidation fold: {0}/{1}'.format(k+1,K1)) 
    
    X_train = X[train_index]
    X_test = X[test_index]
    
    y_train = y[train_index]
    y_test = y[test_index]
    
    N,M = X_train.shape
    
    CV2 = model_selection.KFold(K2, shuffle = True, random_state = 121)
    f = 0
    
    #Defining empty weight array for the weights corresponding to the M
    #attributes as a function of the fold and the value of lambda
    w = np.empty((M+1,K2,len(lambdas)))
    train_error_lr = np.empty((K2,len(lambdas)))
    test_error_lr = np.empty((K2,len(lambdas)))
    
    
    
    for (j,(train_idx, test_idx)) in enumerate(CV2.split(X_train,y_train)):
        
        print('\Inner-Crossvalidation fold: {0}/{1}'.format(j+1,K2)) 
        
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
        -------PHASE 3 - Classification tree--------
        --------------------------------------------
        """
        # Parameters for tree classifier
        # 3 complexity levels was found to be optimal
        #tree depth complexity
        tc = np.arange(3, 15, 1)
        
        cltr_Error_train = np.empty((len(tc),K2))
        cltr_Error_test = np.empty((len(tc),K2))
        
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t, random_state=12)
            dtc = dtc.fit(X_p_train,y_p_train.ravel()) #is sometimes problematic (NaN input)
            y_est_test = dtc.predict(X_p_test)
            y_est_train = dtc.predict(X_p_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = np.sum(y_est_test != y_p_test) / float(len(y_est_test))
            misclass_rate_train = np.sum(y_est_train != y_p_train) / float(len(y_est_train))
            cltr_Error_test[i,j], cltr_Error_train[i,j] = misclass_rate_test, misclass_rate_train
    
        
        
    """
    --------------------------------------------
    -------PHASE 4 - LOGISTIC REGRESSION--------
    --------------------------------------------
    """
    for (j,(train_idx, test_idx)) in enumerate(CV2.split(X_train,y_train)):
        
        print('\Inner-Crossvalidation fold: {0}/{1}'.format(j+1,K1)) 
        
        X_p_train = X_train[train_idx]
        X_p_test = X_train[test_idx]
        
        y_p_train = y_train[train_idx]
        y_p_test = y_train[test_idx]
        
        #N_in = X_p_train.shape[0]
        
        # Standardize the training and test set based on training set moments
        mu_in = np.mean(X_p_train, 0)
        sigma_in = np.std(X_p_train, 0)
        
        # Check if not all values of an attribute are 0, to avoid division by 0
        for i in np.where(sigma_in==0)[0]:
            sigma_in[i] = 1
        
        # Standardizing the data
        X_p_train = (X_p_train - mu_in) / sigma_in
        X_p_test = (X_p_test - mu_in) / sigma_in
        
        lambda_interval = np.logspace(1, 3, 10)
        train_error_rate = np.zeros((len(lambda_interval),K2))
        test_error_rate = np.zeros((len(lambda_interval),K2))
        #coefficient_norm = np.zeros((K1,K2))
        
        for i in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[i], random_state=12)
            
            mdl.fit(X_p_train, y_p_train) 
        
            y_train_est = mdl.predict(X_p_train).T
            y_test_est = mdl.predict(X_p_test).T
            
            train_error_rate[i,j] = np.sum(y_train_est != y_p_train) / len(y_p_train)
            test_error_rate[i,j] = np.sum(y_test_est != y_p_test) / len(y_p_test)
        
    
    #----------------------------Standardizing data ---------------------
    
    mu_in = np.mean(X_train, 0)
    sigma_in = np.std(X_train, 0)
    
    # Check if not all values of an attribute are 0, to avoid division by 0
    for i in np.where(sigma_in==0)[0]:
        sigma_in[i] = 1
    
    # Standardizing the data
    X_train = (X_train - mu_in) / sigma_in
    X_test = (X_test - mu_in) / sigma_in
    
    #------------------------ Tree ------------------------------------------
    
    ##Now that we found the optimal tc for this outer fold,
    #We create a model using that parameter
    tc_opt_idx=np.argmin(cltr_Error_test.mean(1))
    tc_opt=tc[tc_opt_idx]
    
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtcO = tree.DecisionTreeClassifier(criterion='gini', max_depth=tc_opt, random_state = 12)
    dtcO = dtcO.fit(X_train,y_train.ravel()) 
    y_est_test = dtcO.predict(X_test)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test_tr = np.sum(y_est_test != y_test) / float(len(y_p_test))
    cltr_test_error[k] = misclass_rate_test_tr
    
    #-------------------- Logistic Regression ---------------------------------
    
    lambda_opt_idx=np.argmin(test_error_rate.mean(1))
    lambda_opt=lambda_interval[lambda_opt_idx]
    
    mdlO = LogisticRegression(penalty='l2', C=1/lambda_opt, random_state=12)
    mdlO.fit(X_train, y_train)

    y_test_est = mdlO.predict(X_test).T
    misclass_rate_test_lg = np.sum(y_test_est != y_test) / float(len(y_test))
    
    lr_test_error[k] = misclass_rate_test_lg
        
       
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
    Lst_ClTree_test.append((cltr_test_error[k]))
    Lst_lambdas.append(lambda_opt)
    Lst_lr_test.append(lr_test_error[k])
    Lst_tc.append(tc_opt)
    
fig=plt.figure()
plt.plot(np.arange(1,K1+1,1),Lst_ClTree_test,'x-')
plt.plot(np.arange(1,K1+1,1),Lst_lr_test,'o-')
plt.plot(np.arange(1,K1+1,1),Lst_Baseline_test,'k--')
plt.xlabel('Fold number',fontsize = 12)
plt.ylabel('Mean square test error', fontsize = 12)
plt.xticks(np.arange(1,K1+1,1),np.arange(1,K1+1,1))
plt.legend(['Class. Tree','log. reg.','baseline'])
plt.tight_layout()
plt.show()
fig.savefig('3modelsError.pdf') 

print("Classification tree: ", Lst_ClTree_test)
print(" ")
print("tc param: ", Lst_tc)
print(" ")
print("Logistic regression: ", Lst_lr_test)
print(" ")
print("lambda param: ", Lst_lambdas)
print(" ")
print("Baseline: ", Lst_Baseline_test)